
filename = "main-hip-amdgcn-amd-amdhsa-gfx90a.s"
f = pipe(paste0("cat ",filename," | sed -n '/amdhsa[.]kernels/,/\\.\\.\\./p'"))
data = readLines(f)
data = yaml::yaml.load(data)
fun = function(x) as.data.frame(x[sapply(x,length) == 1 & !grepl("^.args", names(x))])
tab = do.call(rbind, lapply(data$amdhsa.kernels, fun))

x = paste("c++filt", tab$.symbol)
tab$name = sapply(x, function(x) readLines(pipe(x)))

#sel = tab$.vgpr_spill_count > 0
sel = rep(TRUE,nrow(tab))
print(tab[sel,c("name", ".sgpr_count", ".vgpr_count", ".vgpr_spill_count")])
