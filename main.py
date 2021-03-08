from parseJson import parseProjectJson

def main():
    objData, relData = parseProjectJson()
    print(objData, "\n", relData)

if __name__=="__main__":
    main()