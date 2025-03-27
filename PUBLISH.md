# Publishing Guide for mcp-server-architect

This document outlines the complete process for building, testing, and publishing new versions of the package.

## Pre-publish Checklist

1. **Update Version Numbers**
   - Update version in `mcp_server_architect/version.py`
   - Update version in `mcp_server_architect/__init__.py`
   - Update version in `pyproject.toml`
   - Make sure all version numbers match

2. **Run Linter and Fix Issues**
   ```bash
   # Run linter
   ruff check .
   
   # Fix lint issues
   ruff check --fix .
   
   # Format code
   ruff format .
   ```

3. **Run Build Verification Test**
   ```bash
   # This runs a comprehensive test with the API key
   ./test_build.sh
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Update version to X.Y.Z and prepare for release"
   ```

## Build and Publish

1. **Build Package**
   ```bash
   # Build wheel and source distributions
   uv build --no-sources
   ```

2. **Verify Build**
   ```bash
   # Check the distributions were created correctly
   ls -la dist/
   ```

3. **Publish to TestPyPI** (Optional but Recommended)
   ```bash
   # Set TestPyPI token
   export UV_PUBLISH_TOKEN=your_testpypi_token
   
   # Publish to TestPyPI
   uv publish --publish-url https://test.pypi.org/legacy/
   
   # Test install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ mcp-server-architect
   ```

4. **Publish to PyPI**
   ```bash
   # Set PyPI token
   export UV_PUBLISH_TOKEN=your_pypi_token
   
   # Publish to PyPI
   uv publish
   ```

5. **Create and Push Git Tag**
   ```bash
   # Create a tag with the version
   git tag -a v$(grep -o '".*"' mcp_server_architect/version.py | sed 's/"//g') -m "Version $(grep -o '".*"' mcp_server_architect/version.py | sed 's/"//g')"
   
   # Push the tag
   git push origin v$(grep -o '".*"' mcp_server_architect/version.py | sed 's/"//g')
   
   # Push the commits
   git push origin main
   ```

6. **Verify Installation**
   ```bash
   # In a fresh environment, verify the package installs correctly
   uvx mcp-server-architect --version
   ```

## Post-publish Tasks

1. **Update Documentation** (if needed)
   - Update any documentation sites
   - Update examples

2. **Announce Release** (if appropriate)
   - Announce in relevant channels
   - Update release notes

## Troubleshooting

If you encounter issues during the publish process:

1. **Build Failures**
   - Check version numbers are consistent
   - Verify dependencies in pyproject.toml
   - Run ./test_build.sh to detect issues

2. **PyPI Upload Failures**
   - Verify you have the correct token set
   - Make sure the version doesn't already exist (you can't overwrite existing versions)

3. **Installation Problems**
   - Test installation in a fresh environment
   - Check logs for missing dependencies