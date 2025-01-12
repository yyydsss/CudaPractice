#include <iostream>

template<typename T>
class my_shared_ptr {
    T a;
    int *count;
    my_shared_ptr(T a) : a(a), count(new int(1)) {}
public:
    my_shared_ptr(const my_shared_ptr &other) : a(other.a), count(other.count) {
        (*count)++;
    }
    ~my_shared_ptr() {
        if (--(*count) == 0) {
            delete count;
            delete a;
        }
    }
    my_shared_ptr &operator=(const my_shared_ptr &other) {
        if (this != &other) {
            if (--(*count) == 0) {
                delete count;
                delete a;
            }
            a = other.a;
            count = other.count;
            (*count)++;
        }
    }
    T &operator*() {
        return a;
    }
    T *operator->() {
        return &a;
    }

};

int main(int, char**){
    std::cout << "Hello, from yangds!\n";
    std::cout << "Hello, from yangds!\n";
}
