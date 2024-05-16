from firedrake import *
from firedrake.adjoint import *

continue_annotation()

tape = get_working_tape()
tape.clear_tape()

#
mesh = UnitSquareMesh(2, 2)

constant = Function(FunctionSpace(mesh, "R", 0)).assign(1.0)
# constant = Constant(1.)
# First component of objective
objective = assemble(constant * dx(mesh))

# Readjust Constant
constant.assign(5.0)
objective += assemble(constant * dx(mesh))

print(f"What the objective should be: {objective}")
blocks = tape.get_blocks()
with stop_annotating():
    for i in tape._bar("Evaluating functional").iter(range(len(blocks))):
        blocks[i].recompute()

# ReducedFunctional can result in a scalar or an assembled 1-form
func_value = objective.block_variable.saved_output
print(f"Tape re-evaluation outputs: {func_value}")
