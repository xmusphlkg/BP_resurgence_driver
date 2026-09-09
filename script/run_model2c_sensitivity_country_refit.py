"""Run and optionally promote one long Model 2C sensitivity country refit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_model2c_sensitivity import sensitivity_specs


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def spec_by_tag(tag: str):
    normalized = tag.strip().lower().replace("-", "_")
    for spec in sensitivity_specs():
        if spec.tag == normalized:
            return spec
    raise SystemExit(f"Unknown registered sensitivity tag: {tag}")


def command_env(args: argparse.Namespace) -> dict[str, str]:
    spec = spec_by_tag(args.target_tag)
    iso3 = args.iso3.strip().upper()
    if iso3 not in spec.countries:
        raise SystemExit(f"{iso3} is not registered for sensitivity tag {spec.tag}.")
    env = os.environ.copy()
    env.update(spec.env)
    env.update(
        {
            "MODEL2C_OUTPUT_TAG": args.source_tag.strip().lower().replace("-", "_"),
            "MODEL2C_CONTACT_MODE": "full",
            "MODEL2C_COUNTRIES": iso3,
            "MODEL2C_REQUIRE_MANUSCRIPT_OUTPUTS": "0",
            "MODEL2C_CHAINS": str(args.chains),
            "MODEL2C_PARALLEL_CHAINS": str(args.parallel_chains),
            "MODEL2C_COUNTRY_WORKERS": "1",
            "MODEL2C_WARMUP": str(args.warmup),
            "MODEL2C_SAMPLES": str(args.samples),
            "MODEL2C_ADAPT_DELTA": str(args.adapt_delta),
            "MODEL2C_MAX_TREEDEPTH": str(args.max_treedepth),
            "MODEL2C_SEED": str(args.seed),
            "MODEL2C_SCENARIO_MAX_DRAWS": str(args.scenario_max_draws),
            "MODEL2C_MIN_MANUSCRIPT_DRAWS": str(args.min_manuscript_draws),
        }
    )
    if args.init_source_tag:
        env["MODEL2C_INIT_SOURCE_TAG"] = args.init_source_tag.strip().lower().replace("-", "_")
    if args.step_size is not None:
        env["MODEL2C_STEP_SIZE"] = str(args.step_size)
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--iso3", required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--parallel-chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--adapt-delta", type=float, default=0.99)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--max-treedepth", type=int, default=16)
    parser.add_argument("--scenario-max-draws", type=int, default=160)
    parser.add_argument("--min-manuscript-draws", type=int, default=2000)
    parser.add_argument(
        "--init-source-tag",
        default="",
        help="Optional existing Model 2C output tag whose posterior draws seed chain initial values.",
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    env = command_env(args)
    iso3 = args.iso3.strip().upper()
    run([str(PYTHON), "script/run_model2c_bayes.py", "--countries", iso3], env)
    run([str(PYTHON), "script/build_model2c_counterfactual_scenarios.py"], env)
    if args.promote:
        run(
            [
                str(PYTHON),
                "script/promote_model2c_tagged_sensitivity_refit.py",
                "--source-tag",
                args.source_tag,
                "--target-tag",
                args.target_tag,
                "--iso3",
                iso3,
            ],
            env,
        )


if __name__ == "__main__":
    main()
