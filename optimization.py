from pulp import LpMaximize, LpProblem, LpVariable

# Create Model
model = LpProblem(name="Maximize_Profit", sense=LpMaximize)

# Variables
A = LpVariable(name="A", lowBound=0)
B = LpVariable(name="B", lowBound=0)

# Objective Function
model += 20*A + 30*B

# Constraints
model += 2*A + B <= 100
model += A + B <= 80

# Solve
model.solve()

# Output
print("Product A =", A.value())
print("Product B =", B.value())
print("Max Profit =", model.objective.value())
