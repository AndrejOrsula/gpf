# Message that contains plane as a geometric primitive
# It is represented in Hessian normal form, i.e. using the equation `a*x + b*y + c*z + d = 0`

uint8 COEFFICIENT_A=0 # X component of the plane's normal (normalized)
uint8 COEFFICIENT_B=1 # Y component of the plane's normal (normalized)
uint8 COEFFICIENT_C=2 # Z component of the plane's normal (normalized)
uint8 COEFFICIENT_D=3 # Constant of the Hessian plane equation

# ID of the plane
uint32 id

float64[4] coefficients
