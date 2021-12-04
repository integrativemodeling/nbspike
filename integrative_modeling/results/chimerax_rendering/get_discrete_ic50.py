import numpy as np

IC50 = {
"RBD-9": 522.00,
"RBD-11": 13.50,
"RBD-15": 5.29,
"RBD-16": 79.20,
"RBD-21": 14.10,
"RBD-22": 100.00,
"RBD-23": 7.31,
"RBD-24": 218.00,
"RBD-29": 9.53,
"RBD-35": 12.30,
"RBD-40": 25.60,
"S1-1": 6.74,
"S1-6": 56.10,
"S1-23": 5.70,
"S1-36": 48.50,
"S1-37": 7.54,
"S1-46": 312.0,
"S1-48": 5.82,
"S1-62": 4.95,
"S1-49": 356.00,
"S2-10": 1718.00,
"S2-40": 1712.00
}

# take -ve log and normalize
IC50_transformed = {k: -np.log10(v) for k, v in IC50.items()}

IC50_transformed_scaled = {}
maxval = max(IC50_transformed.values())
minval = min(IC50_transformed.values())
for k, v in IC50_transformed.items():
    IC50_transformed_scaled[k] = (v - minval) / (maxval - minval)

for k in IC50_transformed:
    v1 = IC50[k]
    v2 = IC50_transformed[k]
    v3 = IC50_transformed_scaled[k]
    s = "%7s, original: %.3f, transformed: %.3f, transformed+scaled: %.3f" % (k, v1, v2, v3)
    print(s)

