"""
Warfarin semi-synthetic stochastic-policy design.

Every other benchmark in this repository intervenes with a DETERMINISTIC policy:
`pi(a | x) = 1{a = a*}`, a single arm or a fixed action shift. The estimator and
the sampler both support a general stochastic `pi`, but nothing demonstrates it.
This design closes that gap on real covariates, and does so with a target policy
chosen to make the case for sampling rather than point estimation.

Construction follows Appendix 14.1.4 of Zenati, Bozkurt and Gretton (2025),
arXiv:2506.02793, on the IWPC warfarin dosing table (PharmGKB; see
data/warfarin/SOURCE.txt):

    X            standardised patient covariates -- demographics, anthropometry,
                 BMI, the eight clinical indications, comorbidities, concomitant
                 medications, smoking, and the VKORC1 / CYP2C9 genotypes that
                 dominate warfarin metabolism.
    TherDose(X)  the recorded stable therapeutic dose, mg/week. Treated as an
                 oracle property of the patient, not as an observed variable.
    A            the administered weekly dose, continuous.
    Y(a, X)      max(|a - TherDose(X)| - tolerance * TherDose(X), 0) + eps,
                 eps ~ N(0, outcome_noise^2): the dosing error outside a
                 +/- tolerance band, i.e. how badly a patient is mis-dosed.

The logging policy is confounded through body mass,

    A | X ~ N(mu_T + sigma_T sqrt(theta) Z_BMI,  sigma_T^2 (1 - theta)),

with mu_T, sigma_T the cohort mean and standard deviation of the therapeutic
dose. Marginally the logged dose then matches the therapeutic-dose distribution,
while a fraction `theta` of its variance is explained by BMI -- and BMI is
genuinely predictive of the therapeutic dose in this cohort (corr about +0.31),
so the logged outcomes are confounded.

Three target policies are provided, in increasing order of interest:

    "shift"             N(mu + delta, ...)        mean moves, shape fixed
    "mean_preserving"   50/50 at -delta, +delta   MEAN FIXED, shape moves
    "shifted_mixture"   50/50 at 0, +delta        both move

`mean_preserving` is the reason this design exists. Its conditional action mean
is exactly the logging mean, so the intervention is invisible to any procedure
that reports only E[Y(pi)] through the action mean; the action law is bimodal
with variance larger by delta^2, the mis-dosing outcome is a nonlinear function
of the action, and the interventional OUTCOME law therefore changes shape. A
sampler sees it; a mean estimator does not.

Because the outcome is scalar and the cost piecewise linear in the action, the
interventional law is available in CLOSED FORM: `interventional_cdf` evaluates
it, so a sampler can be scored without a finite oracle draw.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

from causality_utils.policies import (GaussianPolicy, Policy,
                                      mean_preserving_mixture,
                                      GaussianMixturePolicy)

#: The stability filter of Appendix 14.1.4, "columns 38-39 not NA and the
#: stability flag at column 37 equal to 1". Those positions are ZERO-INDEXED
#: into the canonical 68-column PharmGKB "Subject Data" sheet, where they resolve
#: to the three names below (position 37 is the stable-dose flag, 38 the dose,
#: 39 the INR measured on it); one-indexing lands on the wrong columns, and the
#: zero-indexed reading is the one that matches the prose.
DOSE_COLUMN = "Therapeutic Dose of Warfarin"                       # position 38
STABLE_COLUMN = "Subject Reached Stable Dose of Warfarin"          # position 37
STABLE_INR_COLUMN = "INR on Reported Therapeutic Dose of Warfarin"  # position 39

AGE_COLUMN = "Age"
HEIGHT_COLUMN, WEIGHT_COLUMN = "Height (cm)", "Weight (kg)"
INDICATION_COLUMN = "Indication for Warfarin Treatment"

#: Covariates, grouped exactly as Appendix 14.1.4 lists them. Demographics and
#: the pharmacogenetic markers are categorical and one-hot encoded; everything
#: else is numeric or already a 0/1 flag.
CATEGORICAL_COVARIATES = ("Gender", "Race", "Ethnicity",
                          "CYP2C9 consensus", "VKORC1 -1639 consensus")

#: Comorbidities, smoking, and the single-column medications. A blank cell in
#: this table means "not recorded as present", so these read as 0 rather than
#: dropping the patient -- the appendix drops patients on BMI only.
BINARY_COVARIATES = ("Diabetes",
                     "Congestive Heart Failure and/or Cardiomyopathy",
                     "Valve Replacement",
                     "Aspirin",
                     "Acetaminophen or Paracetamol (Tylenol)",
                     "Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day",
                     "Amiodarone (Cordarone)",
                     "Carbamazepine (Tegretol)",
                     "Phenytoin (Dilantin)",
                     "Rifampin or Rifampicin",
                     "Anti-fungal Azoles",
                     "Herbal Medications, Vitamins, Supplements",
                     "Current Smoker")

#: "statins" and "antibiotics" are single items in the appendix list but several
#: columns in the table; each group becomes one indicator, 1 if any member is.
DRUG_GROUPS = {
    "Statins": ("Simvastatin (Zocor)", "Atorvastatin (Lipitor)",
                "Fluvastatin (Lescol)", "Lovastatin (Mevacor)",
                "Pravastatin (Pravachol)", "Rosuvastatin (Crestor)",
                "Cerivastatin (Baycol)"),
    "Antibiotics": ("Sulfonamide Antibiotics", "Macrolide Antibiotics"),
}

#: The eight clinical indications, coded 1-8 in one semicolon-separated cell
#: ("3; 8", and occasionally "1 or 2"); expanded into eight binary indicators.
N_INDICATIONS = 8

#: Appendix 14.1.4 drops columns with empirical sd below this and patients whose
#: BMI is missing or degenerate.
NEAR_CONSTANT_SD = 0.05
MIN_BMI = 3e-3

#: Optional cap on the therapeutic dose, in mg/week. NOT part of the paper: the
#: appendix filters on a stable INR only, and that filter leaves the implausible
#: tail in place (16 patients above 100 mg/week survive it, three of them at
#: 157.5 / 230 / 315 -- the last is 45 mg/day). TherDose(X) is the reference
#: point of the mis-dosing outcome and sets mu_T and sigma_T, so those rows are
#: visible in the design; `max_dose=100.0` removes them. The default is None,
#: which reproduces the paper.
MAX_THERAPEUTIC_DOSE = None

#: The canonical PharmGKB export and the widely-redistributed CS234 mirror differ in
#: a few column names only; the values are identical. Resolving here means the loader
#: accepts either file. See data/warfarin/SOURCE.txt.
COLUMN_ALIASES = {"Race": ("Race (OMB)", "Race"),
                  "Ethnicity": ("Ethnicity (OMB)", "Ethnicity"),
                  "CYP2C9 consensus": ("CYP2C9 consensus", "Cyp2C9 genotypes")}


def _resolve(frame: pd.DataFrame, column: str) -> str:
    """The name this file actually uses for `column`, following COLUMN_ALIASES."""
    for candidate in COLUMN_ALIASES.get(column, (column,)):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"{column!r} not in this warfarin file "
                   f"(tried {COLUMN_ALIASES.get(column, (column,))})")


def _age_decade(frame: pd.DataFrame) -> pd.Series:
    """Age is banded as '60 - 69'; take the lower edge as a numeric decade."""
    decade = frame[AGE_COLUMN].astype(str).str.extract(r"(\d+)")[0].astype(float)
    return decade.fillna(decade.median())


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    """A recorded 0/1 medication or comorbidity column; blank reads as 0."""
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _indication_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """The multi-code indication cell expanded into N_INDICATIONS binary columns."""
    cell = frame[INDICATION_COLUMN].astype(str)
    return pd.DataFrame(
        {f"Indication {code}": cell.str.contains(rf"(?<!\d){code}(?!\d)",
                                                 regex=True).astype(float)
         for code in range(1, N_INDICATIONS + 1)},
        index=frame.index)


def _covariate_frame(frame: pd.DataFrame, bmi: pd.Series) -> pd.DataFrame:
    """The covariate matrix of Appendix 14.1.4, before any column is dropped."""
    numeric = pd.DataFrame({"Age (decade)": _age_decade(frame),
                            HEIGHT_COLUMN: pd.to_numeric(frame[HEIGHT_COLUMN],
                                                         errors="coerce"),
                            WEIGHT_COLUMN: pd.to_numeric(frame[WEIGHT_COLUMN],
                                                         errors="coerce"),
                            "BMI": bmi})
    flags = pd.DataFrame({name: _flag(frame, name) for name in BINARY_COVARIATES})
    groups = pd.DataFrame({name: np.max([_flag(frame, member) for member in members],
                                        axis=0)
                           for name, members in DRUG_GROUPS.items()},
                          index=frame.index)
    categorical = frame[[_resolve(frame, name) for name in CATEGORICAL_COVARIATES]]
    dummies = pd.get_dummies(categorical.fillna("Unknown"), drop_first=True,
                             dtype=float)
    return pd.concat([numeric, _indication_indicators(frame), flags, groups,
                      dummies], axis=1)


def load_warfarin_cohort(csv_path: str,
                         max_dose: Optional[float] = MAX_THERAPEUTIC_DOSE,
                         near_constant_sd: float = NEAR_CONSTANT_SD,
                         min_bmi: float = MIN_BMI,
                         ) -> dict:
    """
    The stable-dose / stable-INR cohort of Appendix 14.1.4, standardised.

    Three steps, in the appendix's order: (i) keep subjects with the stable-dose
    flag set and both the therapeutic dose and the INR measured on it recorded;
    (ii) build X from demographics, anthropometrics, BMI, the eight clinical
    indications, comorbidities and concomitant medications, smoking, and the
    CYP2C9 / VKORC1 -1639 genotypes; (iii) drop near-constant columns and
    patients with a missing or degenerate BMI. Every column is then centred and
    scaled to unit variance.

    Parameters
    ----------
    csv_path : str
        Path to the IWPC subject table (canonical PharmGKB export or CS234 mirror).
    max_dose : float or None
        Drop patients whose recorded therapeutic dose exceeds this, in mg/week.
        None -- the default, and the paper -- keeps them; see MAX_THERAPEUTIC_DOSE.
    near_constant_sd : float
        Columns with empirical standard deviation below this are removed.
    min_bmi : float
        Patients whose BMI is missing or at or below this are removed.

    Returns
    -------
    dict with keys
        "X"            float64 array (n, d), standardised covariates
        "ther_dose"    float64 array (n,), mg/week
        "bmi_z"        float64 array (n,), standardised BMI (the confounder)
        "feature_names" list of str
        "dropped_features" list of str, the near-constant columns removed
    """
    frame = pd.read_csv(csv_path, low_memory=False)
    dose = pd.to_numeric(frame[DOSE_COLUMN], errors="coerce")
    stable_inr = pd.to_numeric(frame[STABLE_INR_COLUMN], errors="coerce")
    stable = pd.to_numeric(frame[STABLE_COLUMN], errors="coerce")

    keep = (stable == 1) & dose.notna() & stable_inr.notna()
    if max_dose is not None:
        keep &= dose <= max_dose

    height = pd.to_numeric(frame[HEIGHT_COLUMN], errors="coerce")
    weight = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce")
    bmi = weight / (height / 100.0) ** 2
    design = _covariate_frame(frame, bmi)

    # A missing height or weight is exactly a missing BMI, so this one rule is
    # the appendix's only patient-level drop after the stability filter.
    keep &= bmi.notna() & (bmi > min_bmi) & design.notna().all(axis=1)
    if not keep.any():
        raise ValueError("no patients survive the stability and BMI filters")

    design = design.loc[keep]
    spread = design.std(axis=0, ddof=0)
    dropped = spread.index[spread < near_constant_sd].tolist()
    design = design.drop(columns=dropped)

    values = design.to_numpy(float)
    bmi = bmi.loc[keep].to_numpy(float)
    return {"X": (values - values.mean(axis=0)) / values.std(axis=0),
            "ther_dose": dose.loc[keep].to_numpy(float),
            "bmi_z": (bmi - bmi.mean()) / bmi.std(),
            "feature_names": list(design.columns),
            "dropped_features": dropped}


def warfarin_policies(ther_dose: np.ndarray,
                      bmi_z: np.ndarray,
                      theta: float = 0.5,
                      delta_scale: float = 1.0,
                      target: str = "mean_preserving") -> Tuple[Policy, Policy, dict]:
    """
    The confounded logging policy and one of the three target policies.

    The policies are indexed by ROW: `mean_fn` receives a column of row indices,
    not the covariates, because the logging mean depends on that patient's BMI.
    This keeps the `Policy` interface unchanged while letting the mean be an
    arbitrary per-patient number.

    Parameters
    ----------
    ther_dose : array (n,)
        Therapeutic dose in mg/week; sets the marginal scale of the design.
    bmi_z : array (n,)
        Standardised BMI, the confounding channel.
    theta : float in (0, 1)
        Fraction of the logged dose's variance explained by BMI. theta = 0 gives
        an unconfounded design; theta = 1 a degenerate one.
    delta_scale : float
        The target shift delta, as a multiple of the logging policy's CONDITIONAL
        standard deviation sigma_T sqrt(1 - theta). delta_scale = 1 is the
        calibrated default: the mean-preserving target then has exactly sqrt(2)
        times the conditional action spread, with effective sample size about
        65% of n. delta_scale = 2 collapses the overlap (ESS about 7%).
    target : {"mean_preserving", "shift", "shifted_mixture"}

    Returns
    -------
    (logging_policy, target_policy, info) where info records mu_T, sigma_T,
    the conditional scale and delta.
    """
    if not 0.0 <= theta < 1.0:
        raise ValueError("theta must lie in [0, 1)")
    mu_t = float(np.mean(ther_dose))
    sigma_t = float(np.std(ther_dose))
    conditional_scale = sigma_t * np.sqrt(1.0 - theta)
    delta = delta_scale * conditional_scale

    logged_mean = mu_t + sigma_t * np.sqrt(theta) * np.asarray(bmi_z, dtype=float)

    def mean_fn(rows: np.ndarray) -> np.ndarray:
        """`rows` is a column of integer row indices into the cohort."""
        index = np.asarray(rows, dtype=int).reshape(-1)
        return logged_mean[index][:, None]

    logging_policy = GaussianPolicy(mean_fn, conditional_scale)
    if target == "mean_preserving":
        target_policy = mean_preserving_mixture(logging_policy, delta)
    elif target == "shift":
        target_policy = GaussianPolicy(lambda r: mean_fn(r) + delta, conditional_scale)
    elif target == "shifted_mixture":
        target_policy = GaussianMixturePolicy(mean_fn, [0.0, delta], conditional_scale)
    else:
        raise ValueError(f"unknown target policy {target!r}")

    return logging_policy, target_policy, {
        "mu_t": mu_t, "sigma_t": sigma_t,
        "conditional_scale": conditional_scale, "delta": delta}


def _policy_components(policy: Policy,
                       rows: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    A Gaussian (mixture) policy as component means (n_components, n), a shared
    scalar scale, and mixture weights. A plain `GaussianPolicy` is the
    one-component case.
    """
    if isinstance(policy, GaussianMixturePolicy):
        return (policy._component_means(rows)[:, :, 0], float(policy.scale),
                policy.weights)
    means = np.asarray(policy.mean_fn(rows), dtype=float).reshape(1, -1)
    return means, float(policy.scale), np.ones(1)


def _band_probability(y: np.ndarray, ther_dose: np.ndarray, means: np.ndarray,
                      scale: float, weights: np.ndarray,
                      tolerance: float) -> np.ndarray:
    """
    (1/n) sum_i P(a in [T_i(1 - tol) - y, T_i(1 + tol) + y]) under pi(. | x_i),
    evaluated on the grid `y`. This is the cost CDF only for y >= 0: the cost is
    non-negative, so its CDF is zero below, while this band probability is not.
    """
    upper = ther_dose[:, None] * (1.0 + tolerance) + y[None, :]
    lower = ther_dose[:, None] * (1.0 - tolerance) - y[None, :]
    total = np.zeros(y.shape, dtype=float)
    for weight, mean in zip(weights, means):
        total += weight * (ndtr((upper - mean[:, None]) / scale)
                           - ndtr((lower - mean[:, None]) / scale)).mean(axis=0)
    return total


def interventional_cdf(cohort: dict,
                       grid: np.ndarray,
                       target: str = "mean_preserving",
                       theta: float = 0.5,
                       delta_scale: float = 1.0,
                       tolerance: float = 0.1,
                       outcome_noise: float = 0.0,
                       n_quadrature: int = 16) -> np.ndarray:
    """
    The EXACT interventional outcome CDF F(y) = P(Y(pi) <= y) of this design.

    Why it exists
    -------------
    Every other sampler in this repository is scored against a finite oracle
    draw, which carries its own Monte Carlo error: a sampler can then score
    BETTER than the reference and the comparison becomes meaningless. Here the
    outcome is scalar and the cost is piecewise linear in the action, so the law
    has a closed form and no draw is needed.

    The derivation
    --------------
    With cost y(a, x) = max(|a - T_i| - tol * T_i, 0) and T_i the therapeutic
    dose, the event {y(a, x_i) <= y} is exactly {|a - T_i| <= y + tol * T_i},
    i.e. an interval of actions, so for y >= 0

        F(y) = (1/n) sum_i [ Pi_i(T_i + y + tol T_i) - Pi_i(T_i - y - tol T_i) ]

    with Pi_i the action CDF of pi(. | x_i) -- a Gaussian CDF for "shift" and a
    weight-averaged sum of component Gaussian CDFs for the two mixtures. F(y)
    is zero for y < 0, and F(0) is the atom P(the dose lands inside the
    tolerance band), which a positive `outcome_noise` smears out.

    Parameters
    ----------
    cohort : dict
        As returned by `load_warfarin_cohort`; "ther_dose" and "bmi_z" are used.
    grid : array_like
        Points at which to evaluate F. Evaluated as one vectorised block, so the
        working memory is len(grid) * n floats per mixture component.
    target : {"mean_preserving", "shift", "shifted_mixture"}
    theta, delta_scale : float
        Passed to `warfarin_policies`; must match the sample being scored.
    tolerance : float
        Half-width of the acceptable dosing band, as a fraction of T_i.
    outcome_noise : float
        Standard deviation of the observation noise added to the cost. The
        default 0.0 returns the NOISE-FREE law. A positive value convolves F
        with N(0, outcome_noise^2), treating the atom at zero exactly and the
        continuous part by Gauss-Hermite quadrature; use it whenever the sample
        being scored was drawn with the same noise.
    n_quadrature : int
        Gauss-Hermite nodes for that convolution (ignored when the noise is 0).

    Returns
    -------
    np.ndarray of shape (len(grid),)
        F evaluated on `grid`.
    """
    ther_dose = np.asarray(cohort["ther_dose"], dtype=float)
    _, target_policy, _ = warfarin_policies(ther_dose, cohort["bmi_z"], theta,
                                            delta_scale, target)
    rows = np.arange(len(ther_dose), dtype=float)[:, None]
    means, scale, weights = _policy_components(target_policy, rows)

    grid = np.asarray(grid, dtype=float).reshape(-1)
    band = lambda y: np.where(y >= 0.0,
                              _band_probability(np.maximum(y, 0.0), ther_dose,
                                                means, scale, weights, tolerance),
                              0.0)
    if outcome_noise <= 0.0:
        return band(grid)

    # Y = C + eps with C >= 0 and an atom at zero. Splitting that atom off keeps
    # the quadrature on a continuous integrand: the discontinuity is carried
    # exactly by the Gaussian CDF of the noise.
    atom = float(band(np.zeros(1))[0])
    nodes, quad_weights = np.polynomial.hermite.hermgauss(n_quadrature)
    total = atom * ndtr(grid / outcome_noise)
    for node, weight in zip(nodes, quad_weights):
        shifted = grid - np.sqrt(2.0) * outcome_noise * node
        total += (weight / np.sqrt(np.pi)) * np.where(shifted >= 0.0,
                                                      band(shifted) - atom, 0.0)
    return total


def dosing_outcome(actions: np.ndarray,
                   ther_dose: np.ndarray,
                   tolerance: float = 0.1,
                   outcome_noise: float = 0.1,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Mis-dosing cost max(|a - TherDose| - tolerance * TherDose, 0) + noise.

    Zero inside a +/- `tolerance` band around the patient's therapeutic dose and
    growing linearly outside it. The kink is what makes the outcome a nonlinear
    function of the action, so a mean-preserving change in the action law moves
    the outcome law.
    """
    actions = np.asarray(actions, dtype=float).reshape(-1)
    ther_dose = np.asarray(ther_dose, dtype=float).reshape(-1)
    cost = np.maximum(np.abs(actions - ther_dose) - tolerance * ther_dose, 0.0)
    if outcome_noise > 0.0:
        rng = np.random.default_rng() if rng is None else rng
        cost = cost + outcome_noise * rng.standard_normal(cost.shape)
    return cost


def sample_warfarin_data(cohort: dict,
                         n_samples: Optional[int] = None,
                         theta: float = 0.5,
                         delta_scale: float = 1.0,
                         target: str = "mean_preserving",
                         tolerance: float = 0.1,
                         outcome_noise: float = 0.1,
                         seed: int = 0) -> dict:
    """
    One logged sample plus the oracle target-policy draw, in this repo's DGP
    contract (same keys as the dSprites and DecFM designs).

    Patients are drawn WITH replacement when `n_samples` exceeds the cohort
    size, so the sample-size axis can be swept past the cohort; the
    covariates repeat but the actions and outcome noise do not.

    Returns
    -------
    dict with "X", "A_logged", "Y_logged", "A_target", "Y_target",
    "importance_weights", "ther_dose", "logging_policy", "target_policy",
    "policy_info", "row_index".
    """
    rng = np.random.default_rng(seed)
    n_cohort = len(cohort["ther_dose"])
    n_samples = n_cohort if n_samples is None else int(n_samples)
    rows = rng.choice(n_cohort, size=n_samples, replace=n_samples > n_cohort)

    ther_dose = cohort["ther_dose"][rows]
    logging_policy, target_policy, info = warfarin_policies(
        cohort["ther_dose"], cohort["bmi_z"], theta, delta_scale, target)

    row_column = rows[:, None].astype(float)
    a_logged = logging_policy.sample(row_column, rng).reshape(-1)
    a_target = target_policy.sample(row_column, rng).reshape(-1)
    y_logged = dosing_outcome(a_logged, ther_dose, tolerance, outcome_noise, rng)
    y_target = dosing_outcome(a_target, ther_dose, tolerance, outcome_noise, rng)

    log_ratio = (target_policy.log_prob(row_column, a_logged[:, None])
                 - logging_policy.log_prob(row_column, a_logged[:, None]))

    tensor = lambda a, d=torch.float64: torch.as_tensor(np.asarray(a), dtype=d)
    return {"X": tensor(cohort["X"][rows]),
            "A_logged": tensor(a_logged)[:, None],
            "Y_logged": tensor(y_logged)[:, None],
            "A_target": tensor(a_target)[:, None],
            "Y_target": tensor(y_target)[:, None],
            "importance_weights": tensor(np.exp(log_ratio)),
            "ther_dose": tensor(ther_dose),
            "row_index": torch.as_tensor(rows, dtype=torch.long),
            "logging_policy": logging_policy,
            "target_policy": target_policy,
            "policy_info": info}
