def main():
    keywords = ['foo', 'bar', 'bar','1','3','2', 'foo', 'baz', 'foo']
    test = dict(dict.fromkeys(keywords))
    print(test)


if __name__ == '__main__':
    main()
