# vibeyolo top-level Makefile.
#
# Discovers every hw/ip/<block>/dv/Makefile and recurses.

IP_DIRS := $(sort $(dir $(wildcard hw/ip/*/dv/Makefile)))

.PHONY: test lint clean $(IP_DIRS)

test:
	@for d in $(IP_DIRS); do \
	  echo "════ test $$d ════"; \
	  $(MAKE) -C $$d test || exit $$?; \
	done

lint:
	@for d in $(IP_DIRS); do \
	  echo "════ lint $$d ════"; \
	  $(MAKE) -C $$d lint || exit $$?; \
	done

clean:
	@for d in $(IP_DIRS); do $(MAKE) -C $$d clean; done
