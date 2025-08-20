#!/usr/bin/env python3
"""
Fix the license field in the generated pyproject.toml file.
Useful because OpenAPI Generator sometimes sets license = "NoLicense".
"""

from pathlib import Path
import re
import sys

def fix_license_in_pyproject():
    """Fix the license field in the generated pyproject.toml"""
    pyproject_path = Path("build/lex_db_api/pyproject.toml")

    if not pyproject_path.exists():
        print(f"❌ {pyproject_path} not found! Did generation fail?")
        sys.exit(1)

    print(f"🔧 Fixing license in {pyproject_path}")
    
    try:
        # Read the file
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace license field
        pattern = r'license\s*=\s*"NoLicense"'
        replacement = 'license = "MIT"'

        new_content = re.sub(pattern, replacement, content)

        # Write back
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print("✅ License fixed in pyproject.toml")
    except Exception as e:
        print(f"❌ Failed to fix pyproject.toml: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fix_license_in_pyproject()