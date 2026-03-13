import os

def main():
    if os.environ.get("ETK_LIGHT", "0") == "1":
        from echotk.sector_extract_light import main as _main
    else:
        from echotk.sector_extract import main as _main
    _main()