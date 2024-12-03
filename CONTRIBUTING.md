# Contributing to Stonks

We love your input! We want to make contributing to Stonks as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/stonks/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/stonks/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Code Quality Standards

### Python Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting
- Sort imports using [isort](https://pycqa.github.io/isort/)
- Use type hints (checked with MyPy)
- Document functions and classes using docstrings

### JavaScript/TypeScript Code Style
- Use ESLint with our provided configuration
- Follow Prettier formatting guidelines
- Use TypeScript for new code
- Document components and functions

### Testing
- Write unit tests for new features
- Maintain or improve code coverage
- Include integration tests where appropriate
- Test edge cases and error conditions

### Documentation
- Update relevant documentation
- Include docstrings for Python code
- Document API changes
- Add comments for complex logic

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting, missing semicolons, etc.
- refactor: code restructuring
- test: adding tests
- chore: maintenance

Example:
```
feat(auth): add JWT authentication

- Implement JWT token generation
- Add token validation middleware
- Update user routes to use authentication
```

### Pre-commit Checks
Before submitting a PR, ensure:
1. All tests pass
2. Code is formatted correctly
3. Type checking passes
4. No security vulnerabilities
5. Documentation is updated

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stonks.git
cd stonks
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pre-commit install
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run tests:
```bash
pytest
```

## License
By contributing, you agree that your contributions will be licensed under its MIT License.
