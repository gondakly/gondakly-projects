const wrapper = document.querySelector(".sliderWrapper")//this is to select classes from html file const wrapper = document.querySelector(".className")
const menuItem = document.querySelectorAll(".menuItem")

const products = [
    {
      id: 1,
      title: "Air Force",
      price: 119,
      colors: [
        {
          code: "black",
          img: "./img/air.png",
        },
        {
          code: "darkblue",
          img: "./img/air2.png",
        },
      ],
    },
    {
      id: 2,
      title: "Air Jordan",
      price: 149,
      colors: [
        {
          code: "lightgray",
          img: "./img/jordan.png",
        },
        {
          code: "green",
          img: "./img/jordan2.png",
        },
      ],
    },
    {
      id: 3,
      title: "Blazer",
      price: 109,
      colors: [
        {
          code: "lightgray",
          img: "./img/blazer.png",
        },
        {
          code: "green",
          img: "./img/blazer2.png",
        },
      ],
    },
    {
      id: 4,
      title: "Crater",
      price: 129,
      colors: [
        {
          code: "black",
          img: "./img/crater.png",
        },
        {
        code: "lightgray",
        img: "./img/crater2.png",
        },
    ],
    },
    {
    id: 5,
    title: "Hippie",
    price: 99,
    colors: [
        {
        code: "gray",
        img: "./img/hippie.png",
        },
        {
        code: "black",
        img: "./img/hippie2.png",
        },
    ],
    },
];
let choosenProduct = products[0];
const crrentproductimg = document.querySelector(".productImg");
const crrentproductTitle = document.querySelector(".productTitle");
const crrentproductDesc = document.querySelector(".productDesc");
const crrentproductColor = document.querySelector(".color");
const crrentproductSizes= document.querySelectorAll(".Size");
const crrentproductPrice= document.querySelector(".productPrice");
menuItem.forEach((menuItem,index)=>{
    menuItem.addEventListener("click",()=>{
        //change the current slide
        wrapper.style.transform =`translateX(${-100*index}vw)`;// `` important because its a skew function and can be edited
        //wrapper.style.transform & this is just to call a class and make that if we click a slider item the slider wrapper will be transformed by the x-axis
        //`translateX(${-100*index}vw)` by this translation equation

        //change the choosen product
        choosenProduct=products[index]

        //change texts of currentProduct
        crrentproductTitle.textContent = choosenProduct.title;
        crrentproductPrice.textContent = "$"+choosenProduct.price;
        crrentproductimg.src = choosenProduct.colors[0].img
        //assign new colors
        crrentproductColor.forEach((colors,index)=>{
        colors.style.backgroundColor = choosenProduct.colors[index].code;
        });       
    });
});