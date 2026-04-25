from rdkit.Chem import AllChem

params = AllChem.ETKDGv3()

print("Type:", type(params))
print("\nAll public attributes:\n")

for name in sorted(dir(params)):
    if not name.startswith("_"):
        try:
            val = getattr(params, name)
            print(f"{name:30s} -> {type(val)}")
        except Exception as e:
            print(f"{name:30s} -> <error: {e}>")

print("\nSpecific checks:")
for attr in ["maxIterations", "maxIterations", "randomSeed", "useRandomCoords"]:
    print(f"hasattr({attr}):", hasattr(params, attr))
