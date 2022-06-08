################################################################################ 
################## Statistical Programming Languages 2021/22 ###################
##################               Take-home Exam              ###################
##################                                           ###################
##################   	  Sigbert Klinke, Eva-Maria Maier,     ###################
##################       		  Alexander Volkmann             ###################
################################################################################

#-------------------------------------------------------------------------------
# Surname: Lai
# Name: Dingyi
# Student ID (Matrikelnummer): 615865
#-------------------------------------------------------------------------------

### Exercise 5 -----------------------------------------------------------------
rm(list = ls())

getwd()
setwd("/Users/aubrey/Desktop/HU Berlin/SPL")
getwd()

# Read bavarian_nicknames_mal.csv
temp <- readLines("bavarian_nicknames_mal.csv", encoding = "UTF-8")

# Delete leading or tailing white spaces
temp2 <- sapply(temp,trimws)

# Split the data
temp <- strsplit(temp2, "[ ,;.+]+")

# Initiate the list for storing
na <- c()
nina <- c()

# After \t is name!!!

# Store the data
for(i in  1:length(temp)){
        
        # Locate the \t
        spli <- grep("\t", temp[[i]])
        loc <- temp[[i]][spli]
        ind <- regexpr("\t", loc)

        # Control so that no empty lines
        if((length(temp[[i]])!=0) && (grepl("Name",temp[[i]][1])==F)){
                
                # Store name
                if(spli<length(temp[[i]])){
                        temp_na <- c(temp[[i]][(spli+1):length(temp[[i]])],
                                    substr(loc, start = ind+1,stop = 100L))
                }else{
                        temp_na <- c(substr(loc, start = ind+1,stop = 100L))
                }
                
                na <- c(na,temp_na)
                
                # Store nickname
                if(spli>1){
                        nina <- c(nina, rep(paste(paste(temp[[i]][1:(spli-1)],collapse = ", "),
                                                   substr(loc, start = 1,stop = ind-1),
                                                   sep = ", "),length(temp_na)))
                }else{
                        nina <- c(nina, rep(paste(substr(loc, start = 1,stop = ind-1),
                                              sep = ", "),length(temp_na)))
                }
        }
}


# Build a default nick
nick1 <- data.frame(list(Nickname = nina, Name = na))

# All square brackets [] and their containing text are removed
# from the values in the variable Name. But not in Nickname!!!
nick1$Name <- gsub("(\\[.*?\\])","",nick1$Name)

# check leading or tailing white spaces
print(grep("^[ \t]+|[ \t]+$", nick1$Name))

# No empty lines
nick1 <- nick1[!nick1$Name=="",]


# Store unique names
na2 <- unique(nick1$Name)
nina2 <- list()

# For each unique name, merge the corresponding nickname
# Here the nicknames are not unique yet, but will control later
for(j in 1:length(na2)){
        temp_nina2 <- paste(unique(nick1$Nickname[nick1$Name==na2[j]]),collapse = ', ')
        nina2 <- append(nina2, temp_nina2)
}

nick2 <- data.frame(list(Nickname = unlist(nina2), Name = unlist(na2)))

# All shortening notation in the values of Nickname are replaced by a full notation of the alternatives
for(i in 1:nrow(nick2)){
        # Transform "+"
        try <- strsplit(nick2$Nickname[i], "[\t, ;.]+")[[1]]
        in1 <- regexpr("/", try)[1:length(try)]
        in2 <- regexpr("\\(.*?", try)[1:length(try)]
        in3 <- regexpr("\\)", try)[1:length(try)]
        nina2 <- c()
        for(j in 1:length(try)){
                ind1 <- in1[j]
                ind21 <- in2[j]
                ind22 <- in3[j]
                # Transform "/"
                if(ind1 != -1 ){
                        nina2 <- c(nina2,substr(try[j], start = 1,stop = ind1-1),
                                              paste0(substr(try[j], start = 1,stop = ind1-2),
                                                     substr(try[j], start = ind1+1,stop = 100L)))
                }
                # Transform "()"
                if((ind21 != -1)&&(ind22 != -1) ){
                        ori <- paste0(substr(try[j], start = 1,stop = ind21-1),
                                      substr(try[j], start = ind22+1,stop = 100L))
                        alt <- gsub("[\\(\\)]", "",try[j])
                        
                        nina2 <- c(nina2,ori, alt)
                }
                # Else remain
                if((ind1 = -1)&&(ind21 = -1)&&(ind22 = -1)){
                        nina2 <- c(nina2,try[j])
                }
        }
        # Clear redundant elements, control unique nickname
        nina2 <- unique(nina2[!grepl("[////(//)]+", nina2)])
        nick2$Nickname[i] <- paste(nina2,collapse = ", ")
}


# check leading or tailing white spaces
print(grep("^[ \t]+|[ \t]+$", nick2$Name))
print(grep("^[ \t]+|[ \t]+$", nick2$Nickname))

# The alphabetical order of nick is based on the variable Name.
nick <- nick2[order(nick2$Name),]
# The data.frame nick is my result








################################## Redundant Codes:
# # The reason why I implement this is at first glance is that I don't know right after \t is name, 
# # rather, I assumed the last one is name. Hence, there are some names occur in nickname.
# # The following is to extract name from nickname, though I already corrected it in the above.
# # Initialize for additional names hiding in the nickname column
# addna <- list()
# addnina <- list()
# 
# # A thorough search
# for(i in 1:length(temp)){
#         #temp[[i]] <- temp[[i]][!grepl("\\[.*?\\]", temp[[i]])]
#         
#         # Control so that no empty lines
#         if((length(temp[[i]])!=0) && (temp[[i]][1]!="Nick-Name")){
#                 
#                 # Fix the nickname assigned from temp
#                 tem <- temp[[i]][1:(length(temp[[i]])-1)]
#                 for(j in 1:length(na2)){
#                         
#                         # Find out whether there is a name in the nickname
#                         ind <- grepl(na2[j],tem, fixed = TRUE) # Logical
#                         num <- grep(na2[j],tem, fixed = TRUE) # Number
#                         
#                         # If there is a match
#                         if((sum(ind) != 0) && (num>1)){
#                                 
#                                 # Ready for wiping out the name from nickname
#                                 old <- paste0(', ',tem[ind])
#                                 
#                                 # Combine nicknames for the name
#                                 nick2$Nickname[nick2$Name==tem[ind]] <- paste(nick2$Nickname[nick2$Name == tem[ind]],
#                                                                               paste(tem[1:(num-1)],collapse = ", "), sep = ", ")
#                                 
#                                 # Wipe out the name from existential nickname
#                                 nick2$Nickname[grep(tem[ind],nick2$Nickname)] <- gsub(old, "", nick2$Nickname[grep(tem[ind],nick2$Nickname)])
#                                 
#                                 # Check whether there are names after the one I check
#                                 if((length(tem)-num)>0){
#                                         for(t in (num+1):length(tem)){
#                                                 old <- paste0(', ',tem[t])
#                                                 # Wipe out the name from existential nickname
#                                                 nick2$Nickname[grep(tem[t],nick2$Nickname)] <- gsub(old, "", nick2$Nickname[grep(tem[t],nick2$Nickname)])
#                                                 # Add them to the list
#                                                 addna <- append(addna,tem[t])
#                                                 addnina <- append(addnina,paste(tem[1:(num-1)],collapse = ", "))
#                                         }
#                                         
#                                 }
#                         }
#                 }
#         }
# }
# # Combine them to nick2
# nick3 <- rbind(nick2,data.frame(list(Nickname = unlist(addnina), Name = unlist(addna))))