# README


**Project Overview**

- **Purpose**: This repository contains a small machine learning / data analysis project analyzing stroke dataset(s). It includes a Python preprocessing/training script and an R Markdown report that uses `reticulate` to run Python code from the Rmd.
- **Dataset**: `healthcare-dataset-stroke-data.csv` (placed in the project root).

**Files**

- `ML_DataAnalysis_Stroke.Rmd`: R Markdown report that presents the analysis and figures.
- `DecisionRandomTree_Stroke.py`: Python script used for preprocessing, modeling and plotting (also used by the Rmd via `reticulate`).
- `DataAnalysis_Stroke.ipynb`: Jupyter notebook copy/alternate analysis.
- `ML_DataAnalysis_Stroke.html`: Rendered HTML report (example output).

**Quick Setup**

- Recommended: create or identify a single Python environment to be used by `reticulate` so knitting is reproducible.

R (install these packages inside R):

```r
install.packages(c("reticulate", "knitr", "rmarkdown"))
```

Python (packages used by the project):

```bash
# run in Windows cmd or PowerShell; make sure you install into the same Python env reticulate will use
py -3 -m pip install pandas numpy scikit-learn seaborn matplotlib ipython joblib
```

If you use conda, you can install with:

```r
# from R
reticulate::conda_install(envname = "your-env", packages = c('pandas','numpy','scikit-learn','seaborn','matplotlib','ipython','joblib'))
```

**Running the analysis**

- From Python (quick run):

```bash
py -3 DecisionRandomTree_Stroke.py
```

- From R Markdown (knit):

1. At the top of your Rmd set the Python environment explicitly (so reticulate won't probe every knit):

```r
library(reticulate)
# for a specific python executable
reticulate::use_python("C:/path/to/python.exe", required = TRUE)
# or for conda
reticulate::use_condaenv("your-env", required = TRUE)
```

2. Use chunk options and caching to speed repeated knits:

```r
knitr::opts_chunk$set(cache = TRUE, cache.lazy = TRUE, cache.path = "cache/", echo = FALSE, message = FALSE, warning = FALSE)
```

3. For Python plotting chunks, avoid auto-display and save the final image to disk, then include it from R. Example Python chunk (Rmd):

```python
# {python, echo=FALSE, results='hide'}
import matplotlib
matplotlib.use('Agg')  # MUST be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

# build your pairplot / figure
# g = sns.pairplot(...)
# g.fig.suptitle(...)

g.fig.savefig('pairplot_all.png', bbox_inches='tight', dpi=150)
plt.close(g.fig)
```

Then include in Rmd:

```r
knitr::include_graphics('pairplot_all.png')
```

**Notes for contributors**

- Keep heavy preprocessing and model training separated into functions that save their outputs (e.g., `processed_data.rds` or `model.joblib`). That allows knitting the report without retraining every time.
- Use `joblib.dump()` / `joblib.load()` in Python code to persist models and heavy intermediate arrays.


```r
rmarkdown::render('ML_DataAnalysis_Stroke.Rmd')
```
