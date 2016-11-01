# Least Squares Collocation for gravity field estimation

Set of programs for calculating spatial covariances of the data,
and apply the Least Squares Collocation method 
to derive the gravity field elements (i.e. free-air anomaly, geoid height, etc),
using heterogeneous observations from satellite altimetry and shipborne surveys.

Some Python code contains C++ code embedded in it for the performance-critical parts.

The core of the Least Squares Collocation inverse procedure is written in Fortran.
