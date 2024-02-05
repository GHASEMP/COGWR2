Examples
    --------
    #basic model calibration

    >>> import libpysal as ps
    >>> data = ps.io.open(data location)
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>>  y= np.array(data.by_col('LnP')).reshape((-1,1))
    >>> x1 = np.array(data.by_col('age')).reshape((-1,1))
    >>> x2 = np.array(data.by_col('Lnarea_tot')).reshape((-1,1))
    >>> x3 = np.array(data.by_col('floor')).reshape((-1,1))
    >>> x4 = np.array(data.by_col('S')).reshape((-1,1))
    >>> x5 = np.array(data.by_col('P')).reshape((-1,1))
    >>> x6 = np.array(data.by_col('E')).reshape((-1,1))
    >>> x7 = np.array(data.by_col('R')).reshape((-1,1))
    >>> x8 = np.array(data.by_col('Nroom')).reshape((-1,1))
    >>> x9= np.array(data.by_col('Qly')).reshape((-1,1))
    >>> x10 = np.array(data.by_col('Typ')).reshape((-1,1))
    >>> x11 = np.array(data.by_col('G')).reshape((-1,1))
    >>> x12 = np.array(data.by_col('L')).reshape((-1,1))
    >>> x13 = np.array(data.by_col('M')).reshape((-1,1))
    >>> x14 = np.array(data.by_col('QOL')).reshape((-1,1))
    >>> X = np.hstack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14])
    >>> model = GWR(coords, y, X, bw=359, kernel='bisquare')
    >>> results = model.fit()
    >>> print(results.params.shape)
def __init__(self, coords, y, X, bw, kernel='bisquare', constant=True,
                 spherical=False, hat_matrix=False, name_x=None,n_jobs=-1):
"""
Initialize class
"""
def _build_wi(self, i, bw):
       if bw == np.inf:
                  wi = np.ones((self.n))
                  return wi
try:
            wi = Kernel(i, self.coords, bw,
                        function=self.kernel, points=self.points,
                        spherical=self.spherical).kernel
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)
return wi
def _local_fit(self, i):
        """
        Local fitting at location i.
        """
        wi = self._build_wi(i, self.bw).reshape(-1, 1)  #local spatial weights
       betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
       predy = np.dot(self.X[i], betas)[0]
       resid = self.y[i] - predy
       influ = np.dot(self.X[i], inv_xtx_xt[:, i])
        w = 1
        return influ, resid, predy, betas.reshape(-1)
def _compute_betas_gwr(y, x, wi):
"""
    compute MLE coefficients using iwls routine

    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS),
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    """
wi          : array
                  n*1, weights to transform observations from location i in GWR
betas       : array
                  k*1, estimated coefficients
xtx_inv_xt  : array
                  iwls throughout to compute GWR hat matrix
                  [X'X]^-1 X'
xT = (x * wi).T  
xtx = np.dot(xT, x)
xtx_inv_xt = linalg.solve(xtx, xT)
betas = np.dot(xtx_inv_xt, y)
return betas, xtx_inv_xt
# GWR kernel function specifications
import numpy as np
def local_cdist(coords_i, coords, spherical):
    """
    Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.
    """
    if spherical:
        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(
            dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0
        return R * c
    else:
        return np.sqrt(np.sum((coords_i - coords)**2, axis=1))

class Kernel(object):
    """
    GWR kernel function specifications.
    
    """

    def __init__(self, i, data, bw=359, function='bisquare',
                 points=None, spherical=False):

        if points is None:
            self.dvec = local_cdist(data[i], data, spherical).reshape(-1)
        else:
            self.dvec = local_cdist(points[i], data, spherical).reshape(-1)

        self.function = function.lower()
        self.bw = bw
        
        self.kernel = self._kernel_funcs(self.dvec / self.bw)

        if self.function == "bisquare":  #Truncate for bisquare
            self.kernel[(self.dvec >= self.bw)] = 0
def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4 
        zs = self.dvec / self.bw 
       if self.function == 'gaussian':
            return np.exp(-0.5 * (zs)**2)
        elif self.function == 'bisquare':
            return (1 - (zs)**2)**2
        else:
            print('Unsupported kernel function', self.function)
y_bar               : array
                          n*1, weighted mean value of y
offset              : array
                          n*1, the offset variable at the ith location.
                          For Poisson model this term is often the size of
                          the population at risk or the expected size of
                          the outcome in spatial epidemiology; Default is
                          None where Ni becomes 1.0 for all locations
n                      : integer
                         number of observations
def y_bar(self):
        """
        weighted mean of y
        """
         n = self.n
        off = self.offset.reshape((-1, 1))
        arr_ybar = np.zeros(shape=(self.n, 1))
        for i in range(n):
            w_i = np.reshape(self.model._build_wi(i, self.model.bw), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1, 1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
        return arr_ybar
def TSS(self):
        """
        geographically weighted total sum of squares

        Methods: p215, (9.9)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.

        """
         n = self.n
        TSS = np.zeros(shape=(n, 1))
        for i in range(n):
            TSS[i] = np.sum(
                np.reshape(self._build_wi(i, self.bw),
                           (-1, 1)) * (self.y.reshape(
                               (-1, 1)) - self.y_bar[i])**2)
        return TSS
def RSS(self):
        """
        geographically weighted residual sum of squares

        Methods: p215, (9.10)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        n = self.n
        resid = self.resid.reshape((-1, 1))
        RSS = np.zeros(shape=(n, 1))
        for i in range(n):
            RSS[i] = np.sum(
                np.reshape(self._build_wi(i, self.bw),
                           (-1, 1)) * resid**2)
        return RSS
R2 = (self.TSS - self.RSS) / self.TSS
mse = (1/self.n)*self.RSS
return R2 , mse

