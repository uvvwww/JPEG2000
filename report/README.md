# LaTeX Report

- Main file: `report.tex`
- Recommended compiler: `xelatex` (because this report uses `ctexart` for Chinese typesetting)

## Build

```bash
cd report
xelatex report.tex
xelatex report.tex
```

If your environment does not have `ctexart`, install a full TeX distribution (e.g., TeX Live) or switch the document class in `report.tex` to `article` and remove `ctex`-specific settings.

