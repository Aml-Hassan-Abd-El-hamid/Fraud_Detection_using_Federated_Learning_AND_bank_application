package com.Fintech.OnlineBanking.dto;

import java.time.LocalDate;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import org.springframework.format.annotation.DateTimeFormat;

public class UserAccountsRequest {
	
	 @Min(value=16, message = "Card Number should be 16 ")
	 @Max(value=16, message = "Card Number should be 16 ")
	 private long cardNumber ;
	 
	 @DateTimeFormat( pattern="yyyy-MM")
	 private LocalDate cardEnd ;
	
	 @Min(value=4, message = "Card Password should be 4 ")
	 @Max(value=4, message = "Card Password should be 4 ")
	 private long cardPassword ;
	 
	 @Min(value=14, message = "National Id should be 14 ")
	 @Max(value=14, message = "National Id should be 14 ")
	 private Long nationalId;
	
	public Long getNationalId() {
		return nationalId;
	}
	public void setNationalId(Long nationalId) {
		this.nationalId = nationalId;
	}
	
	public long getCardNumber() {
		return cardNumber;
	}
	public void setCardNumber(long cardNumber) {
		this.cardNumber = cardNumber;
	}
	public long getCardPassword() {
		return cardPassword;
	}
	public void setCardPassword(long cardPassword) {
		this.cardPassword = cardPassword;
	}
	public LocalDate getCardEnd() {
		return cardEnd;
	}
	public void setCardEnd(LocalDate cardEnd) {
		this.cardEnd = cardEnd;
	}
	
	

	

	
	
}
