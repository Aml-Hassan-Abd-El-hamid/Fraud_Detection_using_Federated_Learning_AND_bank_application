package com.Fintech.OnlineBanking.controller;
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.Fintech.OnlineBanking.domain.UserCard;
import com.Fintech.OnlineBanking.dto.UserAccountsRequest;
import com.Fintech.OnlineBanking.dto.passwordRequest;
import com.Fintech.OnlineBanking.dto.UsernameRequest;
import com.Fintech.OnlineBanking.domain.Message;
import com.Fintech.OnlineBanking.service.SignUpService;

@RestController
@RequestMapping("/api/v1/auth/")
public class SignUpController {
	@Autowired
	private SignUpService userService;
	
	
	private ch.qos.logback.core.net.SocketConnector.ExceptionHandler exceptionHandler;
	
	@PostMapping("/id_visa_info")
	public ResponseEntity<?> registerNationalVisaNumber( @RequestBody UserAccountsRequest user)
	{
		      Message message = userService.findNationalVisaInfo(user);
				return ResponseEntity.ok(message);
	}
	
	
	
	
	
	
    @PostMapping("/username")
    public ResponseEntity<?> registeruserName( @RequestBody UsernameRequest user)
	  {
		  Message message = userService.vaildUsername(user);
			return ResponseEntity.ok(message);
	   }
	
	@PutMapping("/password")

	public ResponseEntity<?> registerPassword( @RequestBody passwordRequest user
			)
	         {
		        Message message = userService.Password(user);
			return ResponseEntity.ok(message);
		
			}
	
	

	
			


}
