
read_pipe = function(cmd) {
    f = pipe(cmd)
    data = readLines(f)
    close(f)
    data
}

filename = "main-hip-amdgcn-amd-amdhsa-gfx90a.s"
data = read_pipe(paste0("cat ",filename," | sed -n '/amdhsa[.]kernels/,/\\.\\.\\./p'"))
data = yaml::yaml.load(data)
fun = function(x) as.data.frame(x[sapply(x,length) == 1 & !grepl("^.args", names(x))])
tab = do.call(rbind, lapply(data$amdhsa.kernels, fun))

x = paste("c++filt", tab$.symbol)
x = sapply(x, function(x) read_pipe(x))
x = gsub(" *\\[.*\\]$","",x)
tab$name = x

data = read_pipe(paste0("cat ",filename," | grep ScratchSize"))
data = gsub('.*: ','',data)
data = as.integer(data)

tab$ScratchSize = data

tab$TotalBytes = 4*(tab$".sgpr_count" + tab$".vgpr_count") + tab$"ScratchSize"

#sel = tab$.vgpr_spill_count > 0
sel = rep(TRUE,nrow(tab))
options(width = 160)
print(tab[sel,c("name", ".sgpr_count", ".vgpr_count", ".vgpr_spill_count","ScratchSize","TotalBytes")])

