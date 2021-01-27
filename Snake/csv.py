dir = "C:/Users/ssit5/Documents/GitHub/Snake_AI/Snake/"
file = open(dir+"PPO.csv","r")
out = open(dir+"PPOAdjusted.csv","a")
for line in file:
    comma_separated_line = line.split(",")
    x = str(int(comma_separated_line[0])*500)
    y = comma_separated_line[1]
    out.write(x + "," + y)
