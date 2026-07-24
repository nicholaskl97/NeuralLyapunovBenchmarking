# Root directory of the repository. This is used to find the Julia scripts and
# write results under the project tree.
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# Project root passed to Julia via --project. Override this if you want to run
# against a different checkout.
PROJDIR ?= $(ROOT_DIR)
# Julia executable to use.
JULIA ?= julia
# Common Julia invocation used for every trial.
RUN_JULIA = $(JULIA) --project="$(PROJDIR)"

# Experiments that can be targeted directly.
EXPERIMENTS := decrease_condition lyapunov-net_variants sampling_method
# Shared trial groups used across experiments.
GROUPS := undriven controlled neural_policy_search

# Trial names for each shared group. These lists are intentionally short and can
# be extended as new trials are added.
TRIALS_undriven := pendulum_undriven double_pendulum_undriven
TRIALS_controlled := pendulum_controlled double_pendulum_lqr acrobot_lqr quadrotor_planar_lqr quadrotor_3d_lqr
TRIALS_neural_policy_search := pendulum_driven double_pendulum acrobot quadrotor_planar quadrotor_3d

# Helper that returns a trial name only when the corresponding Julia script exists.
define trial_exists
$(if $(wildcard $(1)/scripts/$(2).jl),$(2))
endef

# Expand to all runnable targets for a given experiment by scanning the shared
# groups and picking any trial whose script exists.
define experiment_trial_list
$(foreach group,$(GROUPS),$(foreach trial,$(TRIALS_$(group)),$(if $(call trial_exists,$(1),$(trial)),$(1)/$(trial) )))
endef

# Expand to all runnable targets for a given experiment/group pair.
define group_trial_list
$(foreach trial,$(TRIALS_$(2)),$(if $(call trial_exists,$(1),$(trial)),$(1)/$(trial) ))
endef

# Rule template for running one trial. Each trial writes stdout/stderr directly
# into the results directory for that trial.
define run_trial_rule
.PHONY: $(1)/$(2)
$(1)/$(2): $(1)/scripts/$(2).jl
	@echo "Running $(1)/scripts/$(2).jl"
	@mkdir -p "$(PROJDIR)/$(1)/results/$(2)"
	@$(RUN_JULIA) "$(ROOT_DIR)/$(1)/scripts/$(2).jl" 1>"$(PROJDIR)/$(1)/results/$(2)/$(2).out" 2>"$(PROJDIR)/$(1)/results/$(2)/$(2).err"
endef

# Rule template for an experiment target (for example: make decrease_condition).
define experiment_rule
.PHONY: $(1)
$(1): $(call experiment_trial_list,$(1))
endef

# Rule template for a group target (for example: make decrease_condition/controlled).
define group_rule
.PHONY: $(1)/$(2)
$(1)/$(2): $(call group_trial_list,$(1),$(2))
	@if [ -z "$(strip $(call group_trial_list,$(1),$(2)))" ]; then echo "No $(2) trials are available for $(1) yet."; fi
endef

# Default target: run every experiment.
.PHONY: all
all: $(EXPERIMENTS)

# Usage examples:
#   make all
#   make decrease_condition
#   make decrease_condition/controlled
#   make sampling_method/quadrotor_planar_lqr

# Generate the experiment, group, and trial rules from the shared definitions.
$(foreach exp,$(EXPERIMENTS),$(eval $(call experiment_rule,$(exp))))
$(foreach exp,$(EXPERIMENTS),$(foreach group,$(GROUPS),$(foreach trial,$(TRIALS_$(group)),$(if $(wildcard $(exp)/scripts/$(trial).jl),$(eval $(call run_trial_rule,$(exp),$(trial)))))))
$(foreach exp,$(EXPERIMENTS),$(foreach group,$(GROUPS),$(eval $(call group_rule,$(exp),$(group)))))
