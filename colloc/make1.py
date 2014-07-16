from os import system

### observations: SSG
E = '-E../../norte/ssg/ssg_total_res.txt.err'

### observations: GRAV
G = '-G../../norte/grav/ship/grav_nav_res.txt'

### columns to be loaded from SSG file
e = '-e0,1,2,3,4'

### columns to be loaded from GRAV file
g = '-g0,1,2,3'

### signal to be computed: grav: g; geoid: n
S = '-Sg'

### files with covariances: C_ll, C_mm, C_gg, C_gl, C_nl, C_ng
A = '-A/dados3/fspaolo/DATA/norte/covs/model/cov_ll_res1.txt'
B = '-B/dados3/fspaolo/DATA/norte/covs/model/cov_mm_res1.txt'
C = '-C/dados3/fspaolo/DATA/norte/covs/model/cov_gg_res2.txt'
D = '-D/dados3/fspaolo/DATA/norte/covs/model/cov_gl_res1.txt'
F = '-F/dados3/fspaolo/DATA/norte/covs/model/cov_nl_res1.txt'
H = '-H/dados3/fspaolo/DATA/norte/covs/model/cov_ng_res1.txt'

### variance of the signal to be computed: grav: 230.0; geoid: 0.095
v = '-v230.0'

### diameter (-d) or side (-l) of computation Cell [deg]
c = '-d0.7'    # circular: pi*(d/2)**2
#c = '-d0.5'    # circular: pi*(d/2)**2
#c = '-l0.5'    # square: l**2

### grid to be computed: west,east,south,north 
R = '-R-52,-40,-6,6'              # north 1
#R = '-R-54,-30,-16,-6'             # center 1

### grid resolution (deg)
#I = '-I0.0083333333333333332'   # 0.5m
#I = '-I0.016666666666666666'    # 1.0m
#I = '-I0.025000000000000001'      # 1.5m
I = '-I0.033333333333333333'    # 2.0m

### cell scale factor
s = '-s'

### output file
o = '-o/dados3/fspaolo/DATA/norte/grav/models/grav_modelss_2m_res1.txt.err'

system('python colloc.py %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s' \
       % (E,e,g,S,A,B,C,D,F,H,v,c,R,I,o,s))
