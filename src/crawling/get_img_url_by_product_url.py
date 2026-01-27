import re
from curl_cffi import requests


def get_single_product_image():
    return next(
        iter(
            [
                f"https:{path}" if path.startswith("//") else path
                for path in re.findall(
                    r'\\"image\\":\s*\[\s*\\"([^\\"]+)\\"',
                    requests.get(
                        "https://www.coupang.com/vp/products/9034791497?itemId=26504119443&vendorItemId=93478816329&sourceType=CATEGORY&categoryId=176483",
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
                            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                            "Referer": "https://www.coupang.com/",
                            "Cookie": "x-coupang-target-market=KR; x-coupang-accept-language=ko-KR; PCID=17664634680114475919418; MARKETID=17664634680114475919418; sid=d0170f68140f4305b6d630c289b26f3a3b23b0a0; baby-isWide=wide; bm_ss=ab8e18ef4e; ak_bmsc=1F6440FC19D82C4E415B375C4F9F12BB~000000000000000000000000000000~YAAQheQ1F5Jp9dibAQAAevbj/R7K5YJdHMxFpVXsj8Baf1nv31Mhb4XYirO1zcYST5qmNXQL8i7/yiJ75NOsjjvtI4mXx/SX65y4oJIMBQaauW4evNsTYuWnMrqRV5acGm8gLNPlzahti3+z+6Gwtpj36venm4VRy34nZPCjwxIRNDTgUp+FkewE8v4MWw77Y1CjtSfvudYMaJe+mUgJDl7faA6+14AHwQEuOVu71FjoOtVEJWk7mXiAYt/D2+imiFIxD+cMvSTNy0rme+XgyZsBHFaulSTzx7MlbJbzPRFnubrgCHkQgvvCIAdXei0z9gwGGV/67NyOCAvIkQo8hn3Ti77Id692HcmgIAVapGa9n3jgVKsZiiRCyEDy6BR8OLxF3Z/CX0Wb/gfN0qgzAWBkx3M8TQUxTsBtTu3fDxshGbP+LtU1vXXqmhDvCMMygXgs41HTTGEjejcCc3DUvQ==; _abck=20D7C86C72D94700947923C0DD6ED8CA~0~YAAQheQ1F+x49dibAQAAyW3k/Q+17foViPAK05Bf3+ecUWXygx/7fRT09X+kDNi66CZErE//4XlJndWQe3xbbjCQUKsCtBgohoT3yc4VqvYVvV8wbFHH4E6zTiOd83JKnLafjIsx4F0b98LSVHEVHOkhSulSQmmvSixooOLhY8RHFPusjI3SKg2eviCQEVZUdTvVYd7A0OQ+EodkyTm5cRDxiCB7HPYaLrR4BGgvgJMcQTLNvVAmBQxs+BqVaj9kN7gSahNXjucKSZWbteKFZNIW6I6umP8krOFKUq6RTp08vvJCr1Jj23hRbrmzBye1yPtCf7E0FRDXUMgy8Cs8ffmZHIiLlgLf6I0MAEXvSRLlfbl7hl/UgD8la5AbY+hOEgaAsrdOGDYY2VGPBmX/6H7TVc1F9Py2bHl7pgAupSdYO+MlQx+nGccAh8JUMczKJnX077MggUKIWw57RfJatQCE6c71lP65TAZCY9bTVXPcWtYCW5pj1f35cOiAz7M8lWHqTm09GberUwAHp/C6c5W5pPQK1yVYcCBMl161MtdVz1F+px/OzTsdTD9gq9+Rh6rN7O2G7c60M2kZdGRAbVeihJcDJwaXDruRh8MHp6jyKcSbg7n75TFXaVas+mJ+UkTvV/olL8cbMQe9U60FAvsQnNCi+iYGk6onMQWgW2w1LMU26gGj45LynVJXmf6hPXecM8gVqgcQxIivfJtjzsVxrejOuX+Bul05NcpS2QVy~-1~-1~1769491862~AAQAAAAF%2f%2f%2f%2f%2f2b8JMA24+5lk4PDQaBk9npCr1925da1qpbArdNgzn%2fxWXXD+woA5GFwVl4M0xdmV%2fsMYUDZ5P8OfHbPIWHuvUM%2fjx7WhJuMaiulnHl3uDmWPEtj%2fxg4s7l%2fI1GTL8Bhm75E0S71CPKp1bpxaRMsQJMiRmtD7pT4YlVHcq8fQU0BMi9nO3LQR6cXhM%2fr7n1hhPXfyOKWmkjU7SZ6+Eq2DxI+YbMJWYzDtdVAxasS5oV2bTo%3d~-1; bm_s=YAAQheQ1F+149dibAQAAyW3k/QQiYHAnsk2rzYX3p+MEvZYNsKZzfeDTEsPff5I4ZIasd1NnZkwS46Dui2OalkCUD9Nw+qM1TJmquhHRxVT5HyWKKEyAN5hsF2fpNFaEHMLg9IaDM/EX/Fhv/8DXoDIZvSvoMHcb86blqK2pXOv0wLdm8huHVbsbvqW1skCV/gye+8dK0IPv8RlVMUl2XVLdEoyk84DX9BkU88Wg0+FeWOHcN8rA7bfiVkZsfOkg3KN3wlO4SY1MuiA53aJxDyY3q04lQBlY+WV3wCBfxwrfdzCJGwCwwQSLe4gDv5YNgWY9HA92VYIbvkNeoIK3rdRcJX4P+Y73wJnn2r/D023mWN4TFh35ef6B0afqDGjSIFiftyoaLeqK6iLsOGiPXc0YwekP3N4WFkIo3+sVQbQB2uMDoXfbGwSReGs+Ac38D9JvrTPGj1FCcoOuzR1NJnJf7+2RoB0DFkzxjgl0EACVOlzKG9xZqigrpuWFyeNiEiMfuqswQD8l1ro9oquqaY8AxipzSfmsq9iYGt/Di/O7EZkgi2w4C0Di8aE6N47PGypdazdOF4HI1dQ=; bm_so=A5A67C076341B798DDBA57B6C577199E71ED6BC03DB3291BDA7594BF359A583A~YAAQheQ1F+549dibAQAAyW3k/QZKxCpVCCw5zVyiSzAeCZL6clK72YZiYzHJDolF0JelHSR5u8RgpsaKXKWkXNI+ZiTp5lE4X6lGjRUtO45jxHJwSrJuV4BrsnJflKYlWGReozeALSXTBMTgkxzpHfjQncSQwxu9MG0H/tdcsmvobDRjc2AJh3SuP7z8iLE9esZPbe/baGXZ9CRhAxaKMgGt1enYBrCXwCYQFUB+teU7GzdRhI0wqeennfgdqhpeGY/ck8IugJIlTkCfky3pU8eIeOadFaE7DyZ4RYZZoTztNqbLwjegf19seRaLI184uAilrQWUZZ+qKwIpLC0O0Jzayi1OlcPPyhRreX2HwssPoZ0s+DUdAH140UdTq0XC8kC1gtLL9lZHu2TwskxHideSDmnXN/HSPyOL9SWzhikvdoovE9vQjPde2UKgVQUoLC62U/GZetYPnnhD5Bwa; bm_sv=1E7CE6DFEF00FD6ABF4D51AEA28CC676~YAAQheQ1F+949dibAQAAyW3k/R5g2JfUrHrOApb+IldOKcYpmxCCeeYuc/693NgLNfdOw9Nbz2cZm7Em5sGRKXaazC2uvhCBMeTKyyZz1pUbMnV94/Sb2hjs44v05gRD+QrH3zWB4MSrEzaLqfzxg8VqaKkpt+3q/0fzWmGqycGPzexbocdQb62HjpixtQB6cE++Jf8FYIbjweAOmsM0P9yzwUjESnkfv9Qi6BBSYpowbfh/WM5syB41vcu3OoqSqA==~1; bm_sz=54B1C31EC60E1CD843FF594ED0E9104E~YAAQheQ1F/B49dibAQAAyW3k/R64CbJQglDvgmHXCdWUIWaJkPCJwt8yOLgpRQfOMew8OvFUXL/KGHpLrBqGtjwvh7/R7osvvgIl5Sg2oKxp0Vpu/4Pdl485DFESkNO7Q1oip/3fWNIv5bV1+R7cSUIPAJSZ9NxIK2rDZ6LhJVzqytoCRqezOpZZ4d2eruvrMdyXRhcOQZD9Q2q9MJ9QHV5mebxQ+hWMBp9HzcgOpn2jrOrLF2ejCj9n6br4BTQq6dYr2W0zIwdiPOziBpy0tXlJkRudRMnnUyrtvzv7PipTfUaL6kv/CPZgjJmAFWeUrRDg+wV1nsnLSm8FPC+TeeC1Gqf+7yPbebtBNKC1J5DpoJqzgzA9yZjewlv+nYwkHLXpvAuzmRivPhYiifnPUdGE61+k4wzb1qacyEEAyr9oHQkvdKbw75qsbtxsT26bIKmE1RMSHY6Xd4PdaTXfzZmudQQZFaENJK8vUhLpcKjeHjhcix9djPU+1KlXQX3/mECn1Tnd3GzQsD5D3LrZs1X+ZAixzvn4yW2NPnJeocBu/ttcl8QVsJiR8oB8EvfERdDQ5GR27ixwsSxwcnQ5pGatpCmuBw==~4404547~3355206",
                        },
                        impersonate="chrome124",
                    ).text,
                )
            ]
        ),
        "이미지를 찾을 수 없습니다.",
    )


print(get_single_product_image())
