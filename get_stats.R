
read_pipe = function(cmd) {
    f = pipe(cmd)
    data = readLines(f)
    close(f)
    data
}

read_stats = function(filename) {
    data = read_pipe(paste0("cat ",filename," | sed -n '/amdhsa[.]kernels/,/\\.\\.\\./p'"))
    data = yaml::yaml.load(data)
    fun = function(x) as.data.frame(x[sapply(x,length) == 1 & !grepl("^.args", names(x))])
    tab = do.call(rbind, lapply(data$amdhsa.kernels, fun))
    names(tab) = gsub("^[.]","",names(tab))

    x = paste("c++filt", tab$symbol)
    x = sapply(x, function(x) read_pipe(x))
    x = gsub(" *\\[.*\\]$","",x)
    tab$name = x

    data = read_pipe(paste0("cat ",filename," | grep ScratchSize"))
    data = gsub('.*: ','',data)
    data = as.integer(data)

    if (length(data) == nrow(tab)) {
        tab$ScratchSize = data
        tab$TotalBytes = 4*(tab$"sgpr_count" + tab$"vgpr_count") + tab$"ScratchSize"
    }
    
    tab
}

tab = read_stats(filename = "main-hip-amdgcn-amd-amdhsa-gfx90a.s")

sel = rep(TRUE,nrow(tab))
out_tab = tab[sel,names(tab) %in% c("name", "sgpr_count", "vgpr_count", "vgpr_spill_count","ScratchSize","TotalBytes")]

options(width = 160)
#print(out_tab)

knitr::kable(out_tab)
