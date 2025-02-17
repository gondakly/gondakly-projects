document.getElementById("cashCheckbox").onclick=function(){
    const myCheckBox = document.getElementById("cashCheckbox");
    const myLoclabel = document.getElementById("Cash");

    if(myCheckBox.checked){
     myLoclabel.addEventListener("touchcancel");
    }
    
    }