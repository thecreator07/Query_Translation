def countdown(n):
    while n > 0:
        yield n   # pause here
        n -= 1

for x in countdown(3):  # 3, 2, 1
    print(x)


# Eager: builds a big list
# squares_list = [x*x for x in range(10_000_000)]
# print(squares_list)


# Lazy: computes on demand (Generator)
squares_gen = (x*x for x in range(10_000_000))

print(squares_gen.__next__())

# creating a interator
dict_gen = iter([10,29,23,34,232,2])
print(dict_gen.__next__())
print(next(dict_gen))


total = sum(x*x for x in range(1_000_000))
print(total)








def read_records(path):
    with open(path, encoding="utf-8") as f:   # force utf-8
        for line in f:
            yield line.strip()

def transform_records(records):
    for r in records:
        yield r.upper()

def write_records(recs, path):
    with open(path, "w", encoding="utf-8") as out:
        for r in recs:
            out.write(r + "\n")

write_records(transform_records(read_records("input.txt")), "out.txt")






def averager():
    total = 0
    count = 0
    avg = None
    try:
        while True:
            x = yield avg     # receive x
            total += x
            count += 1
            avg = total / count
    finally:
        print("done")

g = averager()
next(g)          # prime: returns initial avg (None)
g.send(10)       # -> 10.0
g.send(20)       # -> 15.0
g.close()        # prints "done"






def flatten(nested):
    for it in nested:
        if isinstance(it, (list, tuple)):
            yield from flatten(it)
        else:
            yield it
            
            
            
            
            
            
            
            
            
            
def subtask():
    yield 1
    yield 2
    return "done!"   # becomes StopIteration("done!")

def task():
    result = yield from subtask()  # forwards iteration
    yield f"subtask said: {result}"

list(task())  # [1, 2, 'subtask said: done!']            






import asyncio

async def agen():
    for i in range(3):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for x in agen():
        print(x)
    
# asyncio.run(main())