.PHONY: docs

define build_notebooks
	mkdir -p $(2)
	for f in $(1)/*.py; do \
		jupytext --to notebook --execute "$$f" \
		         -o "$(2)/$$(basename "$$f" .py).ipynb"; \
	done
endef

docs:
	$(call build_notebooks,examples/Single_GPs,docs/notebooks/single_gps/)
	$(call build_notebooks,examples/Multivariate_GPs,docs/notebooks/multivariate_gps/)
	$(call build_notebooks,examples/Advanced/Aggregated_Data,docs/notebooks/advanced/aggregated_data)
