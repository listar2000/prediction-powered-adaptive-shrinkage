# Third-party notices

## Robust empirical-Bayes confidence intervals (`ebci`)

Parts of the robust EBCI implementation in
`src/pas/estimators/robust_eb.py` and `src/pas/intervals/robust_eb.py`
are Python translations of the `R/eb.R` and `R/cv.R` routines in Michal
Kolesar's `ebci` package:

- Source repository: https://github.com/kolesarm/ebci
- Copyright: Michal Kolesar
- License: MIT

The translated code is used to reproduce the estimator, finite-sample moment
corrections, and robust critical values described by Armstrong, Kolesar, and
Plagborg-Moller (2022), *Robust Empirical Bayes Confidence Intervals*.

The critical values in `src/pas/intervals/robust_eb.py` are also used by
`src/pas/intervals/double_shrinkage_cis.py`, which implements the
double-shrinkage baseline of Rosenman, Dominici, and Miratrix (2023); that
paper's intervals are themselves built on the same `cva` construction.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
