library(ggplot2)
library(ggtern)
library(tidyverse)
library(viridis)


# utility in pragmatics setting
df_prag_accuracy0 <- read_csv("Plots/3000_VQ/simplex/seed0/pragmatics_accuracy.csv")
colnames(df_prag_accuracy0)[colnames(df_prag_accuracy0) == "accuracy"] ="metric"
df_prag_accuracy1 <- read_csv("Plots/3000_VQ/simplex/seed1/pragmatics_accuracy.csv")
colnames(df_prag_accuracy1)[colnames(df_prag_accuracy1) == "accuracy"] ="metric"
df_prag_accuracy2 <- read_csv("Plots/3000_VQ/simplex/seed2/pragmatics_accuracy.csv")
colnames(df_prag_accuracy2)[colnames(df_prag_accuracy2) == "accuracy"] ="metric"
df_prag_accuracy0 <- df_prag_accuracy0 %>% arrange(Utility, Alpha, Complexity)
df_prag_accuracy1 <- df_prag_accuracy1 %>% arrange(Utility, Alpha, Complexity)
df_prag_accuracy2 <- df_prag_accuracy2 %>% arrange(Utility, Alpha, Complexity)

if (!identical(df_prag_accuracy0[, c("Utility", "Alpha", "Complexity")], df_prag_accuracy1[, c("Utility", "Alpha", "Complexity")]) |
    !identical(df_prag_accuracy1[, c("Utility", "Alpha", "Complexity")], df_prag_accuracy2[, c("Utility", "Alpha", "Complexity")])) {
  stop("Data frames are not aligned. Please check the sorting of columns 'Utility', 'Alpha', 'Complexity'.")
}
df_prag_accuracy <- data.frame(
  metric0 = df_prag_accuracy0$metric,
  metric1 = df_prag_accuracy1$metric,
  metric2 = df_prag_accuracy2$metric
)
df_prag_accuracy$mean <- rowMeans(df_prag_accuracy)
df_prag_accuracy$sd <- apply(df_prag_accuracy, 1, sd)
df_prag_accuracy$Utility <- df_prag_accuracy0$Utility
df_prag_accuracy$Alpha <- df_prag_accuracy0$Alpha
df_prag_accuracy$Complexity <- df_prag_accuracy0$Complexity
head(df_prag_accuracy)



# informativeness in semantics setting
df_lexsem_inf0 <- read_csv("Plots/3000_VQ/simplex/seed0/lexsem_informativeness.csv")
colnames(df_lexsem_inf0)[colnames(df_lexsem_inf0) == "informativeness"] ="metric"
df_lexsem_inf1 <- read_csv("Plots/3000_VQ/simplex/seed1/lexsem_informativeness.csv")
colnames(df_lexsem_inf1)[colnames(df_lexsem_inf1) == "informativeness"] ="metric"
df_lexsem_inf2 <- read_csv("Plots/3000_VQ/simplex/seed2/lexsem_informativeness.csv")
colnames(df_lexsem_inf2)[colnames(df_lexsem_inf2) == "informativeness"] ="metric"
df_lexsem_inf0 <- df_lexsem_inf0 %>% arrange(Utility, Alpha, Complexity)
df_lexsem_inf1 <- df_lexsem_inf1 %>% arrange(Utility, Alpha, Complexity)
df_lexsem_inf2 <- df_lexsem_inf2 %>% arrange(Utility, Alpha, Complexity)

if (!identical(df_lexsem_inf0[, c("Utility", "Alpha", "Complexity")], df_lexsem_inf1[, c("Utility", "Alpha", "Complexity")]) |
    !identical(df_lexsem_inf1[, c("Utility", "Alpha", "Complexity")], df_lexsem_inf2[, c("Utility", "Alpha", "Complexity")])) {
  stop("Data frames are not aligned. Please check the sorting of columns 'Utility', 'Alpha', 'Complexity'.")
}
df_lexsem_inf <- data.frame(
  metric0 = df_lexsem_inf0$metric,
  metric1 = df_lexsem_inf1$metric,
  metric2 = df_lexsem_inf2$metric
)
df_lexsem_inf$mean <- rowMeans(df_lexsem_inf)
df_lexsem_inf$sd <- apply(df_lexsem_inf, 1, sd)
df_lexsem_inf$Utility <- df_lexsem_inf0$Utility
df_lexsem_inf$Alpha <- df_lexsem_inf0$Alpha
df_lexsem_inf$Complexity <- df_lexsem_inf0$Complexity
head(df_lexsem_inf)



# complexity in semantics setting
df_complexity0 <- read_csv("Plots/3000_VQ/seed0/complexity_seed0.csv")
colnames(df_complexity0)[colnames(df_complexity0) == "complexity_test"] ="metric"
df_complexity1 <- read_csv("Plots/3000_VQ/seed1/complexity_seed1.csv")
colnames(df_complexity1)[colnames(df_complexity1) == "complexity_test"] ="metric"
df_complexity2 <- read_csv("Plots/3000_VQ/seed2/complexity_seed2.csv")
colnames(df_complexity2)[colnames(df_complexity2) == "complexity_test"] ="metric"
df_complexity0 <- df_complexity0 %>% arrange(Utility, Alpha, Complexity)
df_complexity1 <- df_complexity1 %>% arrange(Utility, Alpha, Complexity)
df_complexity2 <- df_complexity2 %>% arrange(Utility, Alpha, Complexity)

if (!identical(df_complexity0[, c("Utility", "Alpha", "Complexity")], df_complexity1[, c("Utility", "Alpha", "Complexity")]) |
    !identical(df_complexity1[, c("Utility", "Alpha", "Complexity")], df_complexity2[, c("Utility", "Alpha", "Complexity")])) {
  stop("Data frames are not aligned. Please check the sorting of columns 'Utility', 'Alpha', 'Complexity'.")
}
df_complexity0$metric = df_complexity0$metric 
df_complexity1$metric = df_complexity1$metric 
df_complexity2$metric = df_complexity2$metric 
df_complexity <- data.frame(
  metric0 = df_complexity0$metric,
  metric1 = df_complexity1$metric,
  metric2 = df_complexity2$metric
)
df_complexity$mean <- rowMeans(df_complexity)
df_complexity$sd <- apply(df_complexity, 1, sd)
df_complexity$Utility <- df_complexity0$Utility
df_complexity$Alpha <- df_complexity0$Alpha
df_complexity$Complexity <- df_complexity0$Complexity
head(df_complexity)



# NID in semantics setting
df_nid0 <- read_csv("Plots/3000_VQ/seed0/NID_entropy_count_seed0.csv")
colnames(df_nid0)[colnames(df_nid0) == "NID"] ="metric"
df_nid1 <- read_csv("Plots/3000_VQ/seed1/NID_entropy_count_seed1.csv")
colnames(df_nid1)[colnames(df_nid1) == "NID"] ="metric"
df_nid2 <- read_csv("Plots/3000_VQ/seed2/NID_entropy_count_seed2.csv")
colnames(df_nid2)[colnames(df_nid2) == "NID"] ="metric"
df_nid0 <- df_nid0 %>% arrange(Utility, Alpha, Complexity)
df_nid1 <- df_nid1 %>% arrange(Utility, Alpha, Complexity)
df_nid2 <- df_nid2 %>% arrange(Utility, Alpha, Complexity)

if (!identical(df_nid0[, c("Utility", "Alpha", "Complexity")], df_nid1[, c("Utility", "Alpha", "Complexity")]) |
    !identical(df_nid1[, c("Utility", "Alpha", "Complexity")], df_nid2[, c("Utility", "Alpha", "Complexity")])) {
  stop("Data frames are not aligned. Please check the sorting of columns 'Utility', 'Alpha', 'Complexity'.")
}
df_nid<- data.frame(
  metric0 = df_nid0$metric,
  metric1 = df_nid1$metric,
  metric2 = df_nid2$metric
)
df_nid$mean <- rowMeans(df_nid)
df_nid$sd <- apply(df_nid, 1, sd)
sem <- function(x) sd(x) / sqrt(3)
df_nid$sem <- apply(df_nid, 1, sem)
df_nid$Utility <- df_nid0$Utility
df_nid$Alpha <- df_nid0$Alpha
df_nid$Complexity <- df_nid0$Complexity
head(df_nid)


# lexicon size in semantics setting
df_count0 <- read_csv("Plots/3000_VQ/seed0/NID_entropy_count_seed0.csv")
colnames(df_count0)[colnames(df_count0) == "model_topnames"] ="metric"
df_count1 <- read_csv("Plots/3000_VQ/seed1/NID_entropy_count_seed1.csv")
colnames(df_count1)[colnames(df_count1) == "model_topnames"] ="metric"
df_count2 <- read_csv("Plots/3000_VQ/seed2/NID_entropy_count_seed2.csv")
colnames(df_count2)[colnames(df_count2) == "model_topnames"] ="metric"
df_count0 <- df_count0 %>% arrange(Utility, Alpha, Complexity)
df_count1 <- df_count1 %>% arrange(Utility, Alpha, Complexity)
df_count2 <- df_count2 %>% arrange(Utility, Alpha, Complexity)

if (!identical(df_count0[, c("Utility", "Alpha", "Complexity")], df_count1[, c("Utility", "Alpha", "Complexity")]) |
    !identical(df_count1[, c("Utility", "Alpha", "Complexity")], df_count2[, c("Utility", "Alpha", "Complexity")])) {
  stop("Data frames are not aligned. Please check the sorting of columns 'Utility', 'Alpha', 'Complexity'.")
}
df_count0$metric = df_count0$metric 
df_count1$metric = df_count1$metric 
df_count2$metric = df_count2$metric 
df_count<- data.frame(
  metric0 = df_count0$metric,
  metric1 = df_count1$metric,
  metric2 = df_count2$metric
)
df_count$mean <- rowMeans(df_count)
df_count$sd <- apply(df_count, 1, sd)
df_count$Utility <- df_count0$Utility
df_count$Alpha <- df_count0$Alpha
df_count$Complexity <- df_count0$Complexity
head(df_count)



# plot NID

my_plot <- ggtern(df_nid, 
                  aes(Utility, Alpha, 
                      Complexity, value = mean)) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .1 
  ) +
  scale_fill_gradientn(colors = c( "paleturquoise2", "steelblue2", 
                                   "steelblue4", "black"),
                       limits = c(0.52, 1)) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "NID")+
  theme(axis.text = element_text(size = 25),
        axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 15))
ggsave("Plots/3000_VQ/naming_NID.pdf", plot = my_plot, width = 10, height = 8, dpi = 300)




# plot utility in pragmatics

my_plot <- ggtern(df_prag_accuracy, 
                  aes(Utility, Alpha, Complexity, value = mean)) +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .1 
  ) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  scale_fill_gradientn(colors = c("grey", "yellow", "red", "black"), 
                       limits = c(0.5, 1.0)) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "U")+
  theme(axis.text = element_text(size = 25),
        axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 15)) 
ggsave("Plots/3000_VQ/prag_utility.pdf", plot = my_plot, width = 10, height = 8, dpi = 300)




# plot informativeness in semantics

inf_plot = df_lexsem_inf[df_lexsem_inf$mean > -1, ]

my_plot <- ggtern(inf_plot, 
                  aes(Utility, Alpha, 
                      Complexity, value = mean)) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .1 
  ) +
  scale_fill_gradientn(colors = c("grey", "forestgreen", "darkgreen", "black")) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "I") +
  theme(#axis.text = element_text(size = 25),
    axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 20)) 
ggsave("Plots/3000_VQ/naming_inf.pdf", plot = my_plot, width = 10, height = 8, dpi = 300)



# plot lexicon size 

max_metric <- max(df_count$mean, na.rm = TRUE)
my_plot <- ggtern(df_count, 
                  aes(Utility, Alpha, 
                      Complexity, value = mean)) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .1 
  ) +
  scale_fill_gradientn(colors = c("grey", "cyan", "blue", "navyblue")) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "count") +
  theme(#axis.text = element_text(size = 25),
    axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 20)) 
ggsave("Plots/3000_VQ/naming_count.pdf", plot = my_plot, width = 10, height = 8, dpi = 300)



# plot complexity 

my_plot <- ggtern(df_complexity, 
                  aes(Utility, Alpha, 
                      Complexity, value = mean)) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .1  
  ) +
  scale_fill_gradientn(colors = c( "azure2", "cadetblue1", 
                                   "cadetblue3", "cadetblue4")
  ) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "bits")+
  theme(axis.text = element_text(size = 25),
        axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 15))
ggsave("Plots/3000_VQ/naming_compl.pdf", plot = my_plot, width = 10, height = 8, dpi = 300)




# plot zoom NID 

my_plot <- ggtern(df_nid, 
                  aes(Utility, Alpha, 
                      Complexity, value = mean)) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .0045 
  ) +
  scale_fill_gradientn(colors = c( "paleturquoise2", "steelblue2", 
                                   "steelblue4", "black"),
                       limits = c(0.5, 0.7)) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "NID")+
  
  theme(axis.text = element_text(size = 25),
        axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 15))

C = my_plot + theme_zoom_T(0.02) 
ggsave("Plots/3000_VQ/naming_NID_zoom_T.pdf", plot = C, width = 10, height = 8, dpi = 300)



# plot zoom lexicon size

max_metric <- max(df_count$mean, na.rm = TRUE)
my_plot <- ggtern(df_count, 
                  aes(Utility, Alpha, 
                      Complexity, value = mean)) +
  theme_minimal(base_size = 30) +
  theme_showarrows() +
  geom_hex_tern(
    stat = "hex_tern",
    fun = "mean",
    na.rm = TRUE,
    binwidth = .0045
  ) +
  scale_fill_gradientn(colors = c("grey", "cyan", "blue", "navyblue")) +
  labs( x       = expression(lambda[italic("U")]),
        xarrow  = "",
        y       = expression(lambda[italic("I")]),
        yarrow  = "",
        z       = expression(lambda[italic("C")]),
        zarrow  = "") +
  labs(fill = "count") +
  theme(
    axis.title = element_text(size = 40)) +
  guides(fill = guide_colourbar(barwidth = 1, barheight = 20)) 

C = my_plot + theme_zoom_T(0.02) 
ggsave("Plots/3000_VQ/naming_count_zoom_T.pdf", plot = C, width = 10, height = 8, dpi = 300)


