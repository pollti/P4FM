all: clean build

clean: beautiful
	rm -f documentation.pdf

beautiful:
	rm -f pdfa.xmpi
	rm -f *.aux
	rm -f *.bbl
	rm -f *.bcf
	rm -f *.blg
	rm -f *.log
	rm -f *.out
	rm -f *.run.xml
	rm -f *.synctex.gz
	rm -f *.toc
	rm -f *.xmpdata
	rm -f *.swp

build:
	lualatex documentation.tex
	biber documentation
	lualatex documentation.tex
	lualatex documentation.tex
