import sys

def main():
    line_num = 0
    while True:
        try:
            line = input().split(",")
            line_num += 1
        except EOFError as e:
            break

        if line_num == 1:
            print(",".join(line[:-2]+["good_day"]))
        else:
            print(",".join(line[:-1]))
            if sum(list(map(lambda x: len(x), line))) < len(line):
                continue
            print(",".join(line[:-2]+line[-1:]))

if __name__ == '__main__':
    main()
