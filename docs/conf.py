import os
import sys

sys.path.insert(0, os.path.abspath("../pymllm"))
sys.path.insert(0, os.path.abspath("../"))
autodoc_mock_imports = ["torch"]
project = "MLLM <br>"
version = "2.0.0-beta"
release = "1.0.0"
author = "MLLM Contributors"
copyright = "2024-2025, %s" % author
extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "myst_parser",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
language = "en"
languages = ["en", "zh"]
exclude_patterns = ["build"]
source_suffix = [".md"]
pygments_style = "sphinx"
todo_include_todos = False
# == html settings
html_theme = "furo"
html_static_path = ["_static"]
footer_copyright = "© 2024-2025 MLLM"
footer_note = " "
# html_theme_options = {
#     "light_logo": "img/logo.svg",
#     "dark_logo": "img/logo.png",
# }
header_links = [
    ("Home", "https://github.com/UbiquitousLearning/mllm"),
    ("Github", "https://github.com/UbiquitousLearning/mllm"),
]
html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "UbiquitousLearning",
    "github_repo": "mllm",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
}
