# # from
# # https://stackoverflow.com/questions/19086030/
# # can-pip-or-setuptools-distribute-etc-list-the-license-used-by-each-install

# import pkg_resources
# import prettytable


# def get_pkg_license(pkg):
#     try:
#         lines = pkg.get_metadata_lines("METADATA")
#     except KeyError:
#         lines = pkg.get_metadata_lines("PKG-INFO")

#     for line in lines:
#         if line.startswith("License:"):
#             return line[9:]
#     return "(License not found)"


# def print_packages_and_licenses(pkgs=None):
#     if pkgs is None:
#         pkgs = sorted(pkg_resources.working_set, key=lambda x: str(x).lower())
#     else:
#         pkgs = [pkg_resources.get_distribution(pkg) for pkg in pkgs]
#     t = prettytable.PrettyTable(["Package", "License"])
#     for pkg in pkgs:
#         t.add_row((str(pkg), get_pkg_license(pkg)))
#     print(t)


# if __name__ == "__main__":
#     # print_packages_and_licenses()
#     print_packages_and_licenses(
#         pkgs=[
#             "importlib-metadata",
#             "pandas",
#             "pydotplus",
#             "datatable",
#             "anytree",
#             "matplotlib",
#             "numpy",
#             "dirichlet",
#             "dtreeviz",
#             "torch",
#         ]
#     )
#     # +--------------------------+-----------------------------+------------+
#     # |         Package          |           License           |  MANUAL    |  Fine?
#     # +--------------------------+-----------------------------+------------+
#     # | importlib-metadata 6.5.0 |     (License not found)     | Apache 2.0 |    Yes
#     # |       pandas 2.0.0       |     BSD 3-Clause License    |            |    Yes
#     # |     pydotplus 2.0.2      |           UNKNOWN           | MIT        |    Yes
#     # |     datatable 1.0.0      | Mozilla Public License v2.0 |            |    Yes
#     # |      anytree 2.8.0       |          Apache 2.0         |            |    Yes
#     # |     matplotlib 3.7.1     |             PSF             |            |    Yes
#     # |       numpy 1.24.2       |         BSD-3-Clause        |            |    Yes
#     # |      dirichlet 0.9       |           UNKNOWN           | MIT        |    Yes
#     # |      dtreeviz 2.2.1      |             MIT             |            |    Yes
#     # |     torch 2.0.0+cpu      |            BSD-3            |            |    Yes
#     # +--------------------------+-----------------------------+------------+

#     # Result:
#     # https://choosealicense.com/appendix/
#     # Apache 2.0
#     #   https://choosealicense.com/licenses/apache-2.0/
#     #   Licensed works, modifications, and larger works may be distributed under
#     #   different terms and without source code.
#     # MIT
#     #   https://choosealicense.com/licenses/mit/
#     #   Licensed works, modifications, and larger works may be distributed under
#     #   different terms and without source code.
#     # Mozilla Public License 2.0
#     #   https://choosealicense.com/licenses/mpl-2.0/
#     #   However, a larger work using the licensed work may be distributed under
#     #   different terms and without source code for files added in the larger work.
#     # BSD 3-Clause License
#     #   https://choosealicense.com/licenses/bsd-3-clause/
#     #   A permissive license similar to the BSD 2-Clause License, but with a
#     #   3rd clause that prohibits others from using the name of the copyright holder
#     #   or its contributors to promote derived products without written consent.
#     #   (Similar to MIT)
#     # PSF
#     #   Python Software Foundation License
#     #   https://docs.python.org/3/license.html
#     #   All Python licenses, unlike the GPL, let you distribute a modified version
#     #   without making your changes open source.
