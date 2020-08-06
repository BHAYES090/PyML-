f = open("Countries.txt", "r")

countries = []

for line in f:
    line = line.strip()
    countries.append(line)

f.close()

print(countries)
print(len(countries))

##for country in countries:
##    if country[0] == "T":
##        print(country)
##
##v = open("Countries.txt", "w")
##
##while True:
##    country_name_new = input("Country Name > ")
##    f.write(country_name_new, "\n")
##    if country_name_new == "quit":
##        print("Quitting")
##        break
##
##f.close()



