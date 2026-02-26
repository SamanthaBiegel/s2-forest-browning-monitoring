# Sentinel-2 Forest Browning Monitoring

This project is made using [this template](https://github.com/SwissDataScienceCenter/innovation-cookiecutter).
Next steps include:

 - [x] Create project from the Cookiecutter template.
 - [ ] Create a virtual environment to work in an isolated Python installation.
 - [ ] Install [pre-commit](https://pre-commit.com/) hooks.
 - [ ] Keep either [`.gitlab-ci.yml`](https://docs.gitlab.com/ee/ci/yaml/gitlab_ci_yaml.html) or [`.github/`](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python), according to your Git hosting platform.
 - [ ] Update `authors` and `description`, in `pyproject.toml`.
 - [ ] Add development and installation dependencies in `pyproject.toml`, with permissive version constraints.
 - [ ] Add a `LICENSE` file, if applicable. This is *highly recommended* if the project is open source.
 - [ ] Add a [`CITATION.cff`](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files), to ease citation of your work.
 - [ ] Replace this `README.md` with a proper one. Among others, it must explain the overall context, the installation instructions, a quick start guide, and a repository structure description.


## Getting started

In order to use [pre-commit](https://pre-commit.com/) hooks, they need to be registered:

```sh
pre-commit install
```

It is a good practice to manually invoke hooks after installation, just in case:

```sh
pre-commit run --all-files
```

During development, install pinned dependencies in your virtual environment, including the module itself in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), using:

```sh
pip install -r requirements.txt -e . --group dev
```

New dependencies can be specified directly in `pyproject.toml`; `requirements.txt` will be updated by a pre-commit hook.

Unit tests (using [pytest](https://pytest.org/)) are not executed as a pre-commit hook, to keep the overhead to a minimum. Instead, a CI/CD pipeline is configured to run tests after each commit. You can also execute them locally, manually:

```sh
pytest
```
