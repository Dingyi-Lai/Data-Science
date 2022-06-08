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

### Exercise 6 -----------------------------------------------------------------
rm(list = ls())
# If you run other graphs before, dev.off()
while (!is.null(dev.list()))  dev.off()

# By instinct, I want to build an exit to quit, but it isn't mentioned in the assignment.
# Nonetheless, if you want to quit, then when it asks you to enter "y", you can quit if
# you press Return (or Enter) on your keyboard

########################## Subproblems outsourcing
# Create empty plot
empty_plot <- function(n_row, n_col, data){
        plot.new() # Initialize plot
        plot.window(xlim=c(1,n_col), ylim=c(1,n_row), xaxs="i", yaxs="i")
        plot(data$x,data$y, type="n",xlab="", ylab="",
             panel.first = grid(n_col, n_row),las = 1,main = "Memory",
             xaxt = "n",yaxt = "n",
             ylim = c(0.5, n_row+0.5), xlim = c(0.8, n_col+0.2)) # Create plot without axis
        
        # Define the position and labels of tick marks
        v1_l <- c(1:n_col)
        v2_l <- c(1:n_row)
        
        # I try to make figure more pretty
        #v1_a <- seq(from = 1+0.1, to = n_col-0.1, length.out = n_col)
        #v2_a <- seq(from = 1+0.1, to = n_row-0.1, length.out = n_row)
        
        # Add axis to the plot 
        axis(side = 1, # an integer specifying which side of the plot the axis
             # is to be drawn on. The axis is placed as follows: 1=below, 2=left, 3=above and 4=right.
             at = v1_l, labels = v1_l, tick = F, tck = 1)
        axis(side = 2, # an integer specifying which side of the plot the axis
             # is to be drawn on. The axis is placed as follows: 1=below, 2=left, 3=above and 4=right.
             at = v2_l, labels = v2_l, tick = F,las = 1,  tck = 1) # Adjust direction of writing
}

# Map to style of points from input
map_pc <- function(rrow, rcol,data,rrow_col){
        ind <- data['pair'][(data$x == rrow_col[rrow]) & (data$y == rrow_col[rcol]),]
        return(ind) # Provide ind for dfplot
}

# Check entry
check_range <- function(data, swit=T){
        while(swit == T){ # if swit is still T then repeat the following
                rrow_col <- scan(file = "", what = "list", n = 2, quiet = T) # keep quiet
                if(length( which((data$x == rrow_col[1])& (data$y == rrow_col[2])))==1){
                        swit <- F # turn off when the input is valid
                        break # break when player enter correctly
                }else{
                        print("Card not valid. Again:")
                }
        }
        return(rrow_col)
}

# Each turn
turn <- function(data, dfplot, num_p){
        
        print(paste0("Player ",num_p,", choose your first card (1: row, 2: column)!"))

        rrow_col <- check_range(data) ####
        ind <- map_pc(1, 2, data,rrow_col) # calculate ind based on rrow_col
        
        points(rrow_col[2],rrow_col[1],pch = dfplot[ind,]$pch,
               col = dfplot[ind,]$col, cex = 3/(ncol(data)-1))
        
        print(paste0("Player ",num_p,", choose your second card (1: row, 2: column)!"))
        data_c <- data[which((data$x!=rrow_col[1]) |(data$y!=rrow_col[2])),] # Exclude the first point
                
        rrow_col <- c(rrow_col, check_range(data_c))
        ind <- c(ind, map_pc(3, 4, data,rrow_col)) # store it in an array
        
        points(rrow_col[4],rrow_col[3],pch = dfplot[ind[2],]$pch,
               col = dfplot[ind[2],]$col, cex = 3/(ncol(data)-1))
        
        return(rrow_col)
}

# Check whether move on
# If you press Return(or Enter) on your keyboard, you will stop the program
check_y <- function(swit = T){
        while(swit == T){
                
                ans <- scan(file = "", what = "character", n = 1, quiet = T)
                if(ans == 'y'){
                        swit <- F
                        break # break when player enter "y" correctly
                }else{
                        print("Press [y], when you are ready to move on!")
                }
        }
} # It could be modified for a formal exit though.


# replay plot and reset current_plot
replay_currentp <- function(j, current_plot, rrow_col, data){
        replayPlot(current_plot) # Replay plot
        
        # After confirmation via the y-key, the matching cards are removed from
        # the current playing field and assigned to the respective player
        points(rrow_col[2],rrow_col[1],pch = as.character(j), cex = 3/(ncol(data)-1))
        points(rrow_col[4],rrow_col[3],pch = as.character(j), cex = 3/(ncol(data)-1))
        
        current_plot <- recordPlot() # Reset current_plot
                
        return(current_plot)
}


# check whether n is even and smaller than 208
check_n <- function(n_row,n_col){
        n <- n_row * n_col # Justify valid n
        
        if(n %% 2 != 0){
                stop("n_row * n_col must be an even number.")
        }
        
        if(n > 208){
                stop("Too many cards: n_row * n_col must not exceed 208.")
        }
        
        return(n)
}

# Create original data
build_data <- function(n_row, n_col, n){
        # Create all possible data points for plotting
        x<-data.frame(c(1:n_row))
        y<-data.frame(c(1:n_col))
        data <- merge(x,y,all=TRUE)
        colnames(data) <- c("x","y")

        # The n cards are arranged randomly on the playing field by the function 
        # (invisible for players)
        data['pair'] <- NA
        for(i in 1:(n/2)){
                df <- data[is.na(data['pair']),]
                data[is.na(data['pair']),][sample(nrow(df), 2), 'pair'] <- i
        }

        # The index of dfplot could be linked to 'pair' in data
        return(data)
}


build_dfplot <- function(){
        # Create all possible types for plotting
        pch <- data.frame(c(1:13))
        col <- data.frame(c(1:8))
        dfplot <- merge(pch, col, all=TRUE)
        colnames(dfplot) <- c("pch","col")
        return(dfplot)
}

build_leaderboard <- function(n_player){
        # A data.frame to record the scores
        leaderboard <- data.frame(matrix(rep(0,n_player),ncol = n_player))
        collb <- c() # store for colnames of leaderboard
        for(tn in c(1:n_player)){
                collb <- c(collb, paste0("Player ",tn))
        }
        colnames(leaderboard) <- collb
        return(leaderboard)
}



########################## Main session
memory <- function(n_row = 4, n_col = 13, n_player = 2){
        
        n <- check_n(n_row,n_col)
        
        # The player who begins is selected randomly
        totalp <- sample(c(1:n_player),n_player)
        
        print(paste0('Player ',totalp[1],' starts!'))
        print("In each move you have to choose two cards.")
        
        data <- build_data(n_row, n_col, n)
        
        #dev.off()
        # Default plot
        empty_plot(n_row, n_col,data = data)
        
        # Save current plot
        current_plot <- recordPlot()
        
        # Build dfplot for plot
        dfplot <- build_dfplot()
        
        # Build leaderboard
        leaderboard <- build_leaderboard(n_player)
        
        # A parameter to count the number of remained cards
        remain_cards <- nrow(data)

        # When there are cards remained for the play
        while(remain_cards > 0){
                
                for(i in seq_along(totalp)){
                        # i is index of the number of player
                        # j is the number of player
                        j <- totalp[i]
                        
                        # Reset switch
                        swit <- T
                        while((swit == T)&&(remain_cards > 0)){ # There are still pair of cards on the playing field
                                rrow_col <- turn(data, dfplot, num_p = j) # Gain rrow_col from entry
                                ind1 <- map_pc(1,2,data,rrow_col) # Derive ind1
                                ind2 <- map_pc(3,4,data,rrow_col) # Derive ind2
                                if(i == n_player){ # Identify the next player based on index
                                        next_p <- totalp[1]
                                }else{
                                        next_p <- totalp[i+1]
                                }
                                
                                if(ind1 != ind2){ # If the two selected are not paired
                                        print(paste0("Wrong, Player ",next_p," plays! Press [y], when you are ready to move on!"))
                                        check_y(T) # Check entry for "y" to move on and turn off switch
                                        swit <- F
                                        replayPlot(current_plot) # Replay plot
                                        
                                        # End of this move of the current player
                                        break
                                }else{ # If the two selected are paired
                                        turn_cards <- data.frame(matrix(as.numeric(rrow_col), nrow = 2, byrow = T))
                                        colnames(turn_cards) <- c("x","y")
                                        data <- data[(turn_cards[1,1] != data$x)|(turn_cards[1,2] != data$y),]  # Remain_cards decreases by 2
                                        data <- data[(turn_cards[2,1] != data$x)|(turn_cards[2,2] != data$y),]
                                        
                                        remain_cards <- nrow(data)
                                        if(remain_cards == 0){ # If there are no cards left, end the game
                                                swit <- F # Without confirmation of "y"
                                                leaderboard[j] <-  leaderboard[j]+1
                                                current_plot <- replay_currentp(j, current_plot, rrow_col, data)
                                                break
                                        }else{
                                                print(paste0("Correct, Player ",j," plays again! Current leaderboard:"))
                                                leaderboard[j] <-  leaderboard[j]+1 # Count the score
                                                print(leaderboard) # Present the leaderboard
                                                print("Press [y], when you are ready to move on!")
                                                check_y(T)# Check entry for "y" to move on
                                                current_plot <- replay_currentp(j, current_plot, rrow_col, data)
                                        }
                                }
                        }
                }
                        
        }
        
        # After ending the game, calculate the winner
        winner <- colnames(leaderboard[which.max(leaderboard)])
        
        # Figure out whether the winner is unique
        if(ncol(leaderboard)>1){ # More than one player
                if(apply(leaderboard[ , 1:ncol(leaderboard)], MARGIN=1, function(x) length(unique(x)) < ncol(leaderboard))){
                        # not only one winner, also consider multiple winners (not necessary all)
                        winnername <- colnames(leaderboard[,apply(leaderboard[ , 1:ncol(leaderboard)],
                                                                  MARGIN=1, function(x) x == max(x))])
                        nwinnername <- length(winnername)
                        print(paste("Correct,",paste(winnername[1:(nwinnername-1)],collapse = ", "),"and",
                                    winnername[nwinnername],"are tied winners! Final leaderboard:",
                                    sep = " "))
                }else{ # only one winner
                        print(paste0("Correct, ",winner," wins! Final leaderboard:"))
                }
        }else{ # if only one player
                print(paste0("Correct, ",winner," wins! Final leaderboard:"))
        }
        
        
        print(leaderboard)
}

# Test different cases
#memory(n_row = 2, n_col = 2, n_player = 1)
memory(n_row = 3, n_col = 4, n_player = 3)
memory()




