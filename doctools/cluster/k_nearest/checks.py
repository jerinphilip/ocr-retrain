from .distance import euclid_norm, normalized_euclid_norm

def compare(**kwargs):
	x = kwargs["a"]
	t = kwargs["tolerance"]
	
	eps = kwargs["epsilon"]
	
	if x <= t: 
		return True
	
	
	print(x, t, x-t, abs(x-t), eps)
	return (abs(x - t) <= eps)

def extremas(U, **kwargs):
	minm = kwargs["minimum"]
	maxm = kwargs["maximum"]
	if U < minm:
		minm = U
	if U > maxm:
		maxm = U
	return(minm, maxm)

def checks(P, V):
	n = len(V)
	thresh = 0.5
	mi, mx =0.5, 1.0
	for i in range(n):
		u = V[i]
		# Check 1:
		assert(compare(a=euclid_norm(u), tolerance=1, epsilon=1e-6, minimum=0.5, maximum=0.7))
		for j in range(i+1, n):
			v = V[j]
			
			mi, mx = extremas(normalized_euclid_norm(u-v), minimum=mi, maximum=mx)
			# print(normalized_euclid_norm(u-v),mi, mx, end='')
			if normalized_euclid_norm(u-v) <= thresh:
				print('<%s>  <%s>'%(P[i], P[j]))
				#input()
			# Check 3, subsumes Check 2:
			else:
				pass
				#print("")
			assert(compare(a=normalized_euclid_norm(u - v), tolerance=1, epsilon=1e-6))