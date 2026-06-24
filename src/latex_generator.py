"""
LaTeX compiler module.
Attempts to compile the LaTeX manuscript paper.tex to paper.pdf using pdflatex.
"""

import os
import subprocess
import shutil

def compile_latex(root_dir):
    """
    Attempts to compile paper.tex into paper.pdf in the root_dir.
    """
    tex_path = os.path.join(root_dir, 'paper.tex')
    if not os.path.exists(tex_path):
        print(f"--> [ERROR] LaTeX manuscript file not found at: {tex_path}")
        return False
        
    print("--> Checking for pdflatex compiler...")
    pdflatex_bin = shutil.which("pdflatex")
    
    if pdflatex_bin is None:
        print("--> [WARNING] 'pdflatex' was not found in your system's PATH.")
        print("    The paper.tex file is generated and verified, but PDF compilation is skipped.")
        print("    To compile manually: install TeX Live / MiKTeX and run 'pdflatex paper.tex'")
        return False
        
    print(f"--> Found pdflatex at: {pdflatex_bin}")
    print("--> Compiling LaTeX manuscript (Pass 1 of 2)...")
    
    try:
        # Run pdflatex in root_dir
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("--> [WARNING] pdflatex returned non-zero exit code on Pass 1.")
            
        print("--> Compiling LaTeX manuscript (Pass 2 of 2 to resolve references)...")
        result2 = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        pdf_path = os.path.join(root_dir, 'paper.pdf')
        if os.path.exists(pdf_path):
            print(f"--> [SUCCESS] PDF successfully compiled at: {pdf_path}")
            return True
        else:
            print("--> [ERROR] Compilation finished but paper.pdf was not created.")
            return False
            
    except Exception as e:
        print(f"--> [ERROR] An error occurred during LaTeX compilation: {e}")
        return False
