"""Lightweight dependency smoke check for CI and deployments."""

import importlib

REQUIRED_MODULES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly.express",
    "plotly.graph_objects",
    "sklearn",
    "skimage",
    "matplotlib",
    "seaborn",
    "umap",
    "cv2",
    "xgboost",
    "joblib",
    "SimpleITK",
    "lungmask",
    "totalsegmentator",
]


def main() -> int:
    failures = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            failures.append((module, repr(exc)))

    if failures:
        print("Dependency smoke check failed:")
        for module, error in failures:
            print(f" - {module}: {error}")
        return 1

    print("Dependency smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
