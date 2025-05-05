| ID | Name | Type | Shape | Description | Computed by | Sent to |
|---|---|---|---|---|---|---|
| 0 | log\_mean | nparray | $(G,)$ | For each gene, the mean of the log of the counts across all samples in a center $\overline{\log(Y)}^{(k)}_{g} = \tfrac{1}{n_k}\sum_{i=1}^{n_k}\log(Y^{(k)}_{ig})$. | Each center | Server |
| 0 | n\_samples | int |  | The number of samples in a center $n_k$ for each center $k$. | Each center | Server |
| 1 | global\_log\_mean | nparray | $(G,)$ | The mean of the log of the counts across all samples in all centers $\overline{\log(Y)}_g =\sum_{k=1}^{K}\tfrac{n_k}{n}\overline{\log(Y)}^{(k)}_{g}$. | Server | Center |
| 2 | local\_gram\_matrix | nparray | $(p, p)$ | The gram matrix of the local design matrix $G^{(k)} := (X^{(k)})^{\top}X^{(k)}$. | Each center | Server |
| 2 | local\_features | nparray | $(p, G)$ | $\Phi^{(k)}  := X^{(k)\top} Z^{(k)}$. | Each center | Server |
