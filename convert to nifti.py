str = ""
for i in range(300):
    inp = input().split(sep="—")[-1]
    if (i+1) % 3 == 2:
        inp = " ".join(inp.split())
        str += '"' + inp + '",\n'
print(str)