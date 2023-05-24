from setuptools import find_packages, setup

setup(
    name="cxnli",
    description="Croatian XNLI",
    packages=find_packages(
        include=(
            "src",
            "src.*",
        )
    ),
    include_package_data=True,
)
