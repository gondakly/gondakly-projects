const wrapper = document.querySelector(".sliderWrapper")//this is to select classes from html file const wrapper = document.querySelector(".className")
const menuItem = document.querySelectorAll(".menuItem")
menuItem.forEach((menuItem,index)=>{
    menuItem.addEventListener("click",()=>{
        wrapper.style.transform =`translateX(${-100*index}vw)`;// `` important because its a skew function and can be edited
        //wrapper.style.transform & this is just to call a class and make that if we click a slider item the slider wrapper will be transformed by the x-axis
        //`translateX(${-100*index}vw)` by this translation equation
    });
});