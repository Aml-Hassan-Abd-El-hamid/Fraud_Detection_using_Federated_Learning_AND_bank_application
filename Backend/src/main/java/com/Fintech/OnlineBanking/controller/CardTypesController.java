package com.Fintech.OnlineBanking.controller;

import java.util.Collection;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.Fintech.OnlineBanking.domain.CardType;
import com.Fintech.OnlineBanking.service.CardTypesService;
import com.Fintech.OnlineBanking.user.User;

@RestController
@RequestMapping("/show")
public class CardTypesController {

	@Autowired
	CardTypesService cardTypesService;
	
	@GetMapping("/card_types")
	public ResponseEntity<?>showCardTypes(@AuthenticationPrincipal User user)
	{
		      Collection<CardType> allCardTypes= cardTypesService.showAllCardTypes();
				return ResponseEntity.ok(allCardTypes);
	}
	
}
