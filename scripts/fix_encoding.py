"""Clean all python files in src/ and tests/ to UTF-8 without BOM and LF line endings."""

import os

for root, _, files in os.walk("."):
    if ".venv" in root or "build" in root or "dist" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, "rb") as f:
                    content = f.read()
                # Remove UTF-8 BOM if present
                if content.startswith(b"\xef\xbb\xbf"):
                    content = content[3:]
                # Convert UTF-16 if present
                if content.startswith(b"\xff\xfe") or content.startswith(b"\xfe\xff"):
                    content = content.decode("utf-16").encode("utf-8")
                # Normalize CRLF to LF
                text = content.decode("utf-8", errors="replace").replace("\r\n", "\n")
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            except Exception as e:
                print(f"Error processing {path}: {e}")

print("Cleaned all python files to UTF-8")
