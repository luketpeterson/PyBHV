
#include <iostream>

using namespace std;

int main() {

    for (int value = 0; value < 256; ++value) {

        uint64_t result = 0;

        for (int bit=0; bit < 8; ++bit) {
            if ((value >> bit) & 1) {
                result += 1ULL << (bit*8);
            }
        }

        //NOTE: Inline case-statement performs quite a bit better than a static const table.
        // not sure why, probably something to do with how the table gets packed.

        cout << "case " << value << ": increment = " << result << "; break;\n";

        // if (value % 4 != 3)
        //     cout << result << ", ";
        // else
        //     cout << result << ",\n";
    }


}