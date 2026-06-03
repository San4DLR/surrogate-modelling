def main():
    import torch
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_properties(0).name)
    for i in range(torch.cuda.device_count()):
        print(i)
        print(torch.cuda.get_device_properties(i).name)


if __name__ == "__main__":
    main()