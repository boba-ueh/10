using Pkg
using DataFrames
using XLSX

data=DataFrame(XLSX.readtable("dulieu1_baitap1_excel2010.xlsx","Sheet1"))
