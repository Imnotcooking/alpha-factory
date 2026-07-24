PAPER = paper_a_volatility_not_sufficient
OUTDIR = ../output/pdf

.PHONY: all clean

all:
	mkdir -p $(OUTDIR)
	/opt/homebrew/bin/tectonic $(PAPER).tex --outdir $(OUTDIR)

clean:
	rm -f $(OUTDIR)/$(PAPER).pdf $(OUTDIR)/$(PAPER).log
